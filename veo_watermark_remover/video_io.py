from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    video_codec: str
    has_audio: bool
    audio_codec: str | None


def require_ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe was not found on PATH")
    return executable


def probe_video(path: Path) -> VideoMetadata:
    command = [
        require_ffprobe(), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")
    audio_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    duration = float(
        video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0
    )
    frame_count = int(video_stream.get("nb_frames") or round(duration * fps))
    return VideoMetadata(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        video_codec=str(video_stream.get("codec_name", "unknown")),
        has_audio=audio_stream is not None,
        audio_codec=str(audio_stream.get("codec_name")) if audio_stream else None,
    )

