import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image

from videoseek.utils import call_llm_api, convert_to_free_form_text_representation
from config import general_config


overview_tool = {
    "type": "function",
    "function": {
        "name": "overview",
        "description": "To get a structured video summary for the entire video.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def _sample_hybrid_anomaly_frames(
    vr, total_frames: int, num_frames: int, max_scan_samples: int = 300
) -> np.ndarray:
    """
    Hybrid Frame Sampling: 75% uniform frames + 25% visual anomaly peak frames.
    Guarantees returning `num_frames` sorted frame indices (multiple of 8).
    """
    if total_frames <= num_frames:
        return np.linspace(0, max(0, total_frames - 1), num_frames).astype(int)

    num_uniform = int(round(num_frames * 0.75))
    num_anomaly = num_frames - num_uniform

    uniform_indices = np.linspace(0, total_frames - 1, num_uniform).astype(int)
    selected = list(int(i) for i in uniform_indices)

    scan_count = min(total_frames, max(240, max_scan_samples))
    candidate_indices = np.linspace(0, total_frames - 1, scan_count).astype(int)
    candidate_frames = vr.get_batch(candidate_indices).asnumpy()

    # Downsample to 32x32 for fast anomaly & color salience scoring
    small_frames = np.stack(
        [
            np.array(
                Image.fromarray(frame).resize((32, 32), Image.BILINEAR),
                dtype=np.float32,
            )
            for frame in candidate_frames
        ],
        axis=0,
    )

    m = len(small_frames)
    diff_prev = np.zeros(m, dtype=np.float32)
    diff_next = np.zeros(m, dtype=np.float32)
    if m > 1:
        diff_prev[1:] = np.mean(
            np.abs(small_frames[1:] - small_frames[:-1]), axis=(1, 2, 3)
        )
        diff_next[:-1] = np.mean(
            np.abs(small_frames[:-1] - small_frames[1:]), axis=(1, 2, 3)
        )

    median_frame = np.median(small_frames, axis=0, keepdims=True)
    diff_global = np.mean(np.abs(small_frames - median_frame), axis=(1, 2, 3))

    # Color saturation/channel spread and spatial contrast
    color_spread = np.mean(
        np.max(small_frames, axis=-1) - np.min(small_frames, axis=-1), axis=(1, 2)
    )
    spatial_std = np.std(small_frames, axis=(1, 2, 3))

    diff_score = (
        0.4 * (0.5 * (diff_prev + diff_next) + diff_global)
        + 0.3 * color_spread
        + 0.3 * spatial_std
    )

    sorted_order = np.argsort(diff_score)[::-1]
    min_dist = max(1, total_frames // (num_frames * 4))

    # Pass 1: select highest-scoring anomaly frames not adjacent to already selected frames
    for rank_idx in sorted_order:
        if len(selected) >= num_frames:
            break
        f_idx = int(candidate_indices[rank_idx])
        if all(abs(f_idx - s_idx) >= min_dist for s_idx in selected):
            selected.append(f_idx)

    # Pass 2: relax neighborhood distance if still needing frames
    if len(selected) < num_frames:
        selected_set = set(selected)
        for rank_idx in sorted_order:
            if len(selected) >= num_frames:
                break
            f_idx = int(candidate_indices[rank_idx])
            if f_idx not in selected_set:
                selected.append(f_idx)
                selected_set.add(f_idx)

    # Pass 3: fallback fill from any remaining frames
    if len(selected) < num_frames:
        selected_set = set(selected)
        for f_idx in range(total_frames):
            if len(selected) >= num_frames:
                break
            if f_idx not in selected_set:
                selected.append(f_idx)
                selected_set.add(f_idx)

    return np.sort(np.array(selected[:num_frames], dtype=int))


def execute_overview(config: dict, parameters: dict) -> str:
    """
    Execute the overview tool with optional query-aware and anomaly-sensitive sampling.
    """
    vr = parameters["vr"]
    duration = round(len(vr) / vr.get_avg_fps(), 1)
    subtitles = parameters["subtitles"]
    subtitles_str = convert_to_free_form_text_representation(
        subtitles, content_type="subtitle"
    )

    question = parameters.get("question", "") or ""
    query_profile = parameters.get("query_profile", None)

    enable_anomaly_scan = False
    target_cues = []
    if query_profile is not None:
        if isinstance(query_profile, dict):
            enable_anomaly_scan = bool(query_profile.get("enable_anomaly_scan", False))
            target_cues = query_profile.get("target_cues", []) or []
        else:
            enable_anomaly_scan = bool(
                getattr(query_profile, "enable_anomaly_scan", False)
            )
            target_cues = getattr(query_profile, "target_cues", []) or []

    anomaly_overview_enabled = config.get(
        "anomaly_overview_enabled",
        general_config.get("anomaly_overview_enabled", True),
    )

    num_frames = (
        general_config["frame_sampling_factor"] * general_config["overview_base"]
    )
    if num_frames % 8 != 0:
        # round num_frames to the nearest multiple of 8
        num_frames = int(np.ceil(num_frames / 8.0) * 8)

    # Frame Sampling (Hybrid Uniform + Anomaly Peak vs Standard Uniform)
    total_frames = len(vr)
    if anomaly_overview_enabled and enable_anomaly_scan:
        frame_indices = _sample_hybrid_anomaly_frames(
            vr=vr,
            total_frames=total_frames,
            num_frames=num_frames,
            max_scan_samples=config.get("anomaly_scan_samples", 300),
        )
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    cur_timestamps = np.array(
        [round(frame_indice / vr.get_avg_fps(), 1) for frame_indice in frame_indices],
        dtype=np.float32,
    )
    cur_timestamps_str = ", ".join([f"{t:.1f}s" for t in cur_timestamps])
    frames = vr.get_batch(frame_indices).asnumpy()

    # Resize (short side is 256)
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
    frames = np.stack(resized_frames, axis=0)  # (T, H, W, C)

    # Merge each 8 adjacent frames into a 2x4 grid image: (2H, 4W, C).
    group_size = 8
    num_frames, h, w, c = frames.shape
    cur_timestamps = cur_timestamps.reshape(-1, 2, 4)

    num_groups = num_frames // group_size
    frames = frames.reshape(num_groups, 2, 4, h, w, c)
    frames = frames.transpose(0, 1, 3, 2, 4, 5).reshape(num_groups, 2 * h, 4 * w, c)

    sampling_desc = (
        "The video frames are sampled using hybrid uniform + visual anomaly peak detection."
        if (anomaly_overview_enabled and enable_anomaly_scan)
        else "The video frames are uniformly sampled."
    )

    # OpenAI chat-completions multimodal format: content is a list of typed parts.
    content = [
        {
            "type": "text",
            "text": f"The video segment is located at 0.0s - {duration:.1f}s:\n{sampling_desc}\n",
        }
    ]
    for frame, timestamp in zip(frames, cur_timestamps):
        row_1 = ", ".join([f"{t:.1f}s" for t in timestamp[0]])
        row_2 = ", ".join([f"{t:.1f}s" for t in timestamp[1]])
        timestamp_str = f"[[{row_1}],\n[{row_2}]]"
        img = Image.fromarray(frame)
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
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    query_guidance = ""
    if question or target_cues:
        cues_str = ", ".join([str(c) for c in target_cues]) if target_cues else "N/A"
        query_guidance = (
            f"User Question Context: {question}\n"
            f"Target Visual/OCR Cues: {cues_str}\n"
            "IMPORTANT: In addition to describing each frame (~50 words each), pay special attention to "
            "and explicitly note any visible text (OCR), signs, numbers, distinct colors, objects, or sudden "
            "non-narrative events related to the question or target cues if they appear in any frame.\n\n"
        )

    content.append(
        {
            "type": "text",
            "text": (
                "Video Subtitles:\n"
                f"{subtitles_str}\n\n"
                f"{query_guidance}"
                "Please generate descriptions for each frame in the video. The descriptions should be concise and detailed (~50 words each).\n"
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

    if response is None:
        return "Tool execution failed."

    return "\n\n".join(
        [
            f"{frame['timestamp']}: {frame['description']}"
            for frame in json.loads(response.choices[0].message.content)["frames"]
        ]
    )
