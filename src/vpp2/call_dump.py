"""Save one policy call: the RGB condition, the one-step predicted clip, and the action."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any

import numpy as np
import torch

_LOCK = threading.Lock()
_FORMAT_WRITTEN: set[str] = set()


def dump_root() -> Path | None:
    raw = os.environ.get("VPP2_CALL_DUMP_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _safe(value: Any) -> str:
    text = str(value or "unknown").strip() or "unknown"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return cleaned[:120] or "unknown"


def _chw_uint8_to_hwc(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().numpy()
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"Expected uint8 CHW, got {array.shape}")
    return np.ascontiguousarray(np.transpose(array, (1, 2, 0)))


def _as_hwc_uint8(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        return _chw_uint8_to_hwc(frame)
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {array.shape}")
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int = 4) -> None:
    if not frames:
        raise ValueError(f"No frames to write: {path}")
    height, width = frames[0].shape[:2]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError(f"Frame shapes differ for {path}")
    if width % 2 or height % 2:
        raise ValueError(f"Video size must be even, got {width}x{height}")
    from imageio_ffmpeg import get_ffmpeg_exe

    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
    except BrokenPipeError:
        pass
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr.strip()}")


def _ensure_format_note(root: Path) -> None:
    key = str(root)
    if key in _FORMAT_WRITTEN:
        return
    note = root / "format.json"
    if not note.exists():
        note.write_text(
            json.dumps(
                {
                    "alignment": (
                        "current_step is the 0-based simulator observation index. "
                        "The saved simulator video frame with that index is the current observation."
                    ),
                    "input_mp4": "Anchor frame, then 8 history frames ending at the current frame. Playback fps is 4, not simulator time.",
                    "pred_mp4": (
                        "17 frames from one denoising step. The first 9 cover the condition window "
                        "and are not copies of the input. The last 8 are the future. Playback fps is 4."
                    ),
                    "action": "float16 end-effector chunk, shape [horizon, 16].",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    _FORMAT_WRITTEN.add(key)


def save_call(
    *,
    root: Path,
    evaluation_id: Any,
    trial_id: Any,
    env_idx: int,
    episode_index: int,
    call_index: int,
    current_step: int,
    input_steps: list[int],
    input_frames_chw: list[torch.Tensor],
    predicted_frames: list[Any],
    action: np.ndarray,
    ws_inference_step: Any,
) -> None:
    episode_dir = (
        root
        / _safe(evaluation_id)
        / _safe(trial_id)
        / f"env{int(env_idx):02d}"
        / f"episode{int(episode_index):04d}"
    )
    stem = f"call{int(call_index):04d}_step{int(current_step):06d}"
    input_name = f"{stem}_input.mp4"
    pred_name = f"{stem}_pred.mp4"
    action_array = np.asarray(action, dtype=np.float16)
    record = {
        "episode_index": int(episode_index),
        "call_index": int(call_index),
        "env_idx": int(env_idx),
        "current_step": int(current_step),
        "input_steps": [int(step) for step in input_steps],
        "ws_inference_step": None if ws_inference_step is None else int(ws_inference_step),
        "input_video": input_name,
        "pred_video": pred_name,
        "pred_frames": len(predicted_frames),
        "condition_rgb_frames": len(input_frames_chw),
        "action_dtype": "float16",
        "action_shape": list(action_array.shape),
        "action": action_array.tolist(),
    }
    with _LOCK:
        episode_dir.mkdir(parents=True, exist_ok=True)
        _ensure_format_note(root)
        _write_mp4(episode_dir / input_name, [_chw_uint8_to_hwc(frame) for frame in input_frames_chw])
        _write_mp4(episode_dir / pred_name, [_as_hwc_uint8(frame) for frame in predicted_frames])
        with (episode_dir / "calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
