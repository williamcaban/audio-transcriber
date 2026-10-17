"""Turn an uploaded audio/video file into the 16kHz mono wav the ASR model expects.

Shells out to ffmpeg/ffprobe rather than a Python wrapper library so video
containers (mp4, mkv, mov, ...) work the same as plain audio files with zero
extra dependencies beyond ffmpeg itself.
"""

import subprocess


class MediaError(RuntimeError):
    pass


def probe_duration_seconds(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed: {result.stderr.strip()}")
    return float(result.stdout.strip())


def extract_audio(input_path: str, output_wav_path: str) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-vn",
            output_wav_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise MediaError(f"ffmpeg failed: {result.stderr.strip()[-2000:]}")
