"""ffmpeg wrapper: video -> numbered JPEGs at a configurable fps (spec §4).

Output files are %06d.jpg starting at 000001; frame k's timestamp is
(k - 1) / fps after video start.
"""

import subprocess
from pathlib import Path


def extract_frames(
    video_path: Path, out_dir: Path, fps: float, jpeg_quality: int = 2
) -> list[Path]:
    """Extract frames; returns the sorted list of written JPEG paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(jpeg_quality),
        str(out_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted(out_dir.glob("*.jpg"))
