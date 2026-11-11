import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image

from config import general_config
from videoseek.utils import call_llm_api, convert_to_free_form_text_representation


_DEFAULT_MAX_INSPECT_FRAMES = general_config.get(
    "max_inspect_frames",
    general_config.get("frame_sampling_factor", 4)
    * general_config.get("overview_base", 16),
)

inspect_tool = {
    "type": "function",
    "function": {
        "name": "inspect",
        "description": (
            "Unified multi-granular video inspection tool. Inspects any video segment "
            "[start_time, end_time] at a specified sampling rate `fps` to answer or localize `query`. "
            f"Must obey the Frame Budget Rule: (end_time - start_time) * fps <= {_DEFAULT_MAX_INSPECT_FRAMES}."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific visual question or target cue to inspect in the selected video segment.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time (in seconds) of the video segment to inspect.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time (in seconds) of the video segment to inspect.",
                },
                "fps": {
                    "type": "number",
                    "description": (
                        "Sampling rate in frames per second (FPS). Use low FPS (e.g. <= 64 / duration) for whole-video "
                        "macro scans, moderate FPS (e.g. 0.2 - 1.0) for segment localization, or high FPS (4 or 8) "
                        "for dense inspection of short clips (<= 6s)."
                    ),
                },
            },
            "required": ["query", "start_time", "end_time", "fps"],
            "additionalProperties": False,
        },
    },
}


def _parse_macro_json_frames(raw_content: str) -> str:
    """Safely parse JSON frame descriptions returned in Macro/Grid mode."""
    if not raw_content:
        return "No relevant content found."
    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("frames"), list):
            return "\n\n".join(
                [
                    f"{frame.get('timestamp', 'N/A')}: {frame.get('description', '')}"
                    for frame in parsed["frames"]
                ]
            )
    except Exception:
        pass
    return raw_content


def execute_inspect(config: dict, parameters: dict) -> str:
    """
    Execute the unified `inspect` tool with Frame Budget Guardrail and Adaptive Image Packing:
      - Macro / Grid Mode (num_frames > 16): 2x4 grid packing with Timestamp Matrix.
      - Micro / High-Res Mode (num_frames <= 16): Individual high-resolution frames with sub-second timestamps.
    """
    query = str(parameters["query"])
    start_time = float(parameters["start_time"])
    end_time = float(parameters["end_time"])
    fps = float(parameters["fps"])
    vr = parameters["vr"]
    subtitles = parameters["subtitles"]

    avg_fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration = round(total_frames / avg_fps, 2) if avg_fps > 0 else 0.0

    # Normalize time boundaries: 0 <= start_time < end_time <= duration
    was_invalid_window = end_time <= start_time
    start_time = max(0.0, start_time)
    if duration > 0 and start_time >= duration:
        start_time = max(0.0, duration - 1.0)
    if end_time <= start_time:
        end_time = start_time + 1.0
    if duration > 0 and end_time > duration:
        end_time = duration
        if start_time >= end_time or (was_invalid_window and (end_time - start_time) < 1.0):
            start_time = max(0.0, end_time - 1.0)

    if fps <= 0:
        fps = 1.0

    window_duration = max(0.5, end_time - start_time)

    # Frame Budget Guardrail
    max_inspect_frames = int(
        config.get(
            "max_inspect_frames",
            general_config.get(
                "max_inspect_frames",
                general_config.get("frame_sampling_factor", 4)
                * general_config.get("overview_base", 16),
            ),
        )
    )
    desired_frames = max(1, int(round(window_duration * fps)))
    budget_note = ""
    if desired_frames > max_inspect_frames:
        budget_note = (
            f"[System Auto-Adjustment: Clamped from {desired_frames} to {max_inspect_frames} "
            "frames to respect frame budget]\n"
        )
        num_frames = max_inspect_frames
    else:
        num_frames = desired_frames

    subtitles_str = convert_to_free_form_text_representation(
        [
            subtitle
            for subtitle in subtitles
            if float(subtitle["start_time"]) <= float(end_time)
            and float(subtitle["end_time"]) >= float(start_time)
        ],
        content_type="subtitle",
    )

    start_frame = max(0, min(int(start_time * avg_fps), total_frames - 1))
    end_frame = max(
        start_frame + 1, min(int(round(end_time * avg_fps)), total_frames)
    )
    available_frames = max(1, end_frame - start_frame)

    effective_fps = num_frames / window_duration

    # =========================================================================
    # MODE 1: MACRO / GRID MODE (num_frames > 16) -> 2x4 Grid Image Packing
    # =========================================================================
    if num_frames > 16:
        if num_frames % 8 != 0:
            num_frames = int(np.ceil(num_frames / 8.0) * 8)
        if num_frames > max_inspect_frames:
            num_frames = max(16, (max_inspect_frames // 8) * 8)

        frame_indices = np.linspace(
            start_frame, max(start_frame, end_frame - 1), num_frames
        ).astype(int)
        cur_timestamps = np.array(
            [round(frame_idx / avg_fps, 1) for frame_idx in frame_indices],
            dtype=np.float32,
        )
        cur_timestamps_str = ", ".join([f"{t:.1f}s" for t in cur_timestamps])
        frames = vr.get_batch(frame_indices).asnumpy()

        # Resize shorter side to 256px
        _, height, width, _ = frames.shape
        short_side = min(height, width)
        scale = 256 / short_side
        target_height = max(1, int(round(height * scale)))
        target_width = max(1, int(round(width * scale)))

        resized_frames = [
            np.array(
                Image.fromarray(frame).resize(
                    (target_width, target_height),
                    Image.BICUBIC,
                )
            )
            for frame in frames
        ]
        frames = np.stack(resized_frames, axis=0)

        # Merge each 8 adjacent frames into a 2x4 grid image: (2H, 4W, C)
        group_size = 8
        num_frames, h, w, c = frames.shape
        cur_timestamps_grid = cur_timestamps.reshape(-1, 2, 4)
        num_groups = num_frames // group_size
        frames = frames.reshape(num_groups, 2, 4, h, w, c)
        frames = frames.transpose(0, 1, 3, 2, 4, 5).reshape(
            num_groups, 2 * h, 4 * w, c
        )

        content = [
            {
                "type": "text",
                "text": (
                    f"The video segment is located at {start_time:.1f}s - {end_time:.1f}s:\n"
                    f"The video frames are uniformly sampled ({num_frames} frames in 2x4 grids).\n"
                ),
            }
        ]
        for grid_frame, timestamp in zip(frames, cur_timestamps_grid):
            row_1 = ", ".join([f"{t:.1f}s" for t in timestamp[0]])
            row_2 = ", ".join([f"{t:.1f}s" for t in timestamp[1]])
            timestamp_str = f"[[{row_1}],\n[{row_2}]]"
            img = Image.fromarray(grid_frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="jpeg")
            byte_data = output_buffer.getvalue()
            base64_image = base64.b64encode(byte_data).decode("utf-8")
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[{timestamp[0, 0]:.1f}s - {timestamp[-1, -1]:.1f}s]:\n"
                        f"Timestamp Matrix:\n{timestamp_str}\n"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": (
                    "Video Subtitles:\n"
                    f"{subtitles_str}\n\n"
                    f"Inspection Query:\n{query}\n\n"
                    "Please generate descriptions for each frame in the video segment (~45 words each), "
                    "and explicitly highlight any visual details, objects, actions, or text related to the Inspection Query.\n"
                    f"Ensure every timestamp value exactly matches a timestamp from the provided timestamp matrices (same values and formatting): [{cur_timestamps_str}].\n"
                    "Return ONLY valid JSON. Use this exact schema:\n"
                    '{"frames": [{"timestamp": "1.0s", "description": "FRAME_DESCRIPTION_1"}, ...]}\n'
                ),
            }
        )

        response = call_llm_api(
            messages=[{"role": "user", "content": content}],
            model_name=config["model_name"],
            api_base=config["api_base"],
            api_key=config["api_key"],
            api_version=config["api_version"],
            max_tokens=config["max_tokens"],
            reasoning_effort=config["reasoning_effort"],
            seed=config["seed"],
            temperature=config["temperature"],
            return_json=True,
        )
        if response is None or not getattr(response, "choices", None):
            return "Tool execution failed."
        parsed_text = _parse_macro_json_frames(
            response.choices[0].message.content
        )
        return f"{budget_note}{parsed_text}"

    # =========================================================================
    # MODE 2: MICRO / HIGH-RES MODE (num_frames <= 16) -> Individual Full Frames
    # =========================================================================
    num_frames = min(num_frames, available_frames, 16)
    frame_indices = np.linspace(
        start_frame, max(start_frame, end_frame - 1), num_frames
    ).astype(int)

    use_subsecond = fps >= 4 or effective_fps >= 4
    precision = 2 if use_subsecond else 1
    cur_timestamps = np.array(
        [
            round(frame_idx / avg_fps, precision)
            for frame_idx in frame_indices
        ],
        dtype=np.float32,
    )
    frames = vr.get_batch(frame_indices).asnumpy()

    content = [
        {
            "type": "text",
            "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s):\n",
        }
    ]
    for frame, timestamp in zip(frames, cur_timestamps):
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_image = base64.b64encode(byte_data).decode("utf-8")
        ts_str = f"{timestamp:.2f}s" if use_subsecond else f"{timestamp:.1f}s"
        content.append({"type": "text", "text": ts_str})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "Video Subtitles:\n"
                f"{subtitles_str}\n\n"
                f"Question:\n{query}\n\n"
                "Please answer the question and describe fine visual details with exact timestamps based on the given video clip. "
                "If the clip is not related to the question, please return 'No relevant content found.'"
            ),
        }
    )

    response = call_llm_api(
        messages=[{"role": "user", "content": content}],
        model_name=config["model_name"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        api_version=config["api_version"],
        max_tokens=config["max_tokens"],
        reasoning_effort=config["reasoning_effort"],
        seed=config["seed"],
        temperature=config["temperature"],
    )
    if response is None or not getattr(response, "choices", None):
        return "Tool execution failed."
    return f"{budget_note}{response.choices[0].message.content}"
