import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videoseek.utils import call_llm_api, convert_to_free_form_text_representation
from config import general_config


focus_tool = {
    "type": "function",
    "function": {
        "name": "focus",
        "description": (
            "To verify fine visual details, dense inspection of a short clip "
            "(start_time - end_time, <= {focus_num_frames}s, with dynamic FPS 1, 4, or 8)."
        ).format(
            focus_num_frames=general_config["frame_sampling_factor"]
            * general_config["focus_base"]
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to focus the video. The query should be a concise question that can be answered by the video.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the video to focus.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the video to focus.",
                },
                "fps": {
                    "type": "number",
                    "description": (
                        "Sampling rate in frames per second (FPS) to inspect the video clip. "
                        "Choose 1 for general actions (clip <= 16s), 4 for fast movements or fine details (clip <= 6s), "
                        "or 8 for brief events, flashing text/OCR, or sub-second anomalies (clip <= 3s)."
                    ),
                },
            },
            "required": ["query", "start_time", "end_time", "fps"],
            "additionalProperties": False,
        },
    },
}


def _resolve_dynamic_fps(parameters: dict, duration_window: float) -> float:
    """Resolve sampling FPS from tool parameters, query_profile, or clip duration."""
    fps = parameters.get("fps", None)
    try:
        if fps is not None and float(fps) > 0:
            return float(fps)
    except (TypeError, ValueError):
        pass

    query_profile = parameters.get("query_profile", None)
    if query_profile is not None:
        rec_fps = getattr(query_profile, "recommended_fps", None)
        if rec_fps is None and isinstance(query_profile, dict):
            rec_fps = query_profile.get("recommended_fps", None)
        try:
            if rec_fps is not None and float(rec_fps) > 0:
                return float(rec_fps)
        except (TypeError, ValueError):
            pass

    if duration_window <= 3.0:
        return 8.0
    elif duration_window <= 6.0:
        return 4.0
    elif duration_window <= 12.0:
        return 2.0
    return 1.0


def execute_focus(config: dict, parameters: dict) -> str:
    """
    Execute the focus tool with dynamic FPS sampling.
    """
    query = parameters["query"]
    start_time = float(parameters["start_time"])
    end_time = float(parameters["end_time"])
    vr = parameters["vr"]
    subtitles = parameters["subtitles"]

    avg_fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    video_duration = total_frames / avg_fps if avg_fps > 0 else 0.0

    # Safely handle invalid, negative, out-of-bounds, or zero-width time windows
    was_invalid_window = start_time >= end_time
    if was_invalid_window:
        center = max(0.0, start_time)
        if video_duration > 0:
            center = min(center, video_duration)
        start_time = max(0.0, center - 0.5)
        end_time = start_time + 1.0

    start_time = max(0.0, start_time)
    if video_duration > 0:
        if end_time > video_duration:
            end_time = video_duration
        if start_time >= end_time or (was_invalid_window and (end_time - start_time) < 1.0):
            start_time = max(0.0, end_time - 1.0)

    duration_window = max(0.5, float(end_time) - float(start_time))
    fps = _resolve_dynamic_fps(parameters, duration_window)

    subtitles_str = convert_to_free_form_text_representation(
        [
            subtitle
            for subtitle in subtitles
            if float(subtitle["start_time"]) <= float(end_time)
            and float(subtitle["end_time"]) >= float(start_time)
        ],
        content_type="subtitle",
    )

    # Frame Sampling within [0, len(vr) - 1]
    start_frame = max(0, min(int(start_time * avg_fps), total_frames - 1))
    end_frame_exclusive = max(
        start_frame + 1, min(int(round(end_time * avg_fps)), total_frames)
    )

    desired_frames = max(1, int(round(duration_window * fps)))
    max_cap = config.get(
        "max_focus_frames_cap", general_config.get("max_focus_frames_cap", 24)
    )
    available_frames = max(1, end_frame_exclusive - start_frame)
    num_frames = min(desired_frames, available_frames, max_cap)

    frame_indices = np.linspace(
        start_frame, max(start_frame, end_frame_exclusive - 1), num_frames
    ).astype(int)

    precision = 2 if fps >= 4 else 1
    cur_timestamps = np.array(
        [round(frame_indice / avg_fps, precision) for frame_indice in frame_indices],
        dtype=np.float32,
    )
    frames = vr.get_batch(frame_indices).asnumpy()

    content = [
        {
            "type": "text",
            "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s, sampled at {fps:g} FPS):\n",
        }
    ]
    for frame, timestamp in zip(frames, cur_timestamps):
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_image = base64.b64encode(byte_data).decode("utf-8")
        ts_str = f"{timestamp:.2f}s" if fps >= 4 else f"{timestamp:.1f}s"
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
                "Please answer the question based on the given video clip. "
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

    if response is None:
        return "Tool execution failed."

    return response.choices[0].message.content
