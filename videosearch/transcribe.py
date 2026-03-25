"""Transcription via whisper.cpp (local, free).

Uses the `transcribe` script from ~/ygg/bin/ which wraps whisper-cli.
Falls back to a direct whisper-cli call if the script isn't found.
"""

import os
import subprocess
from pathlib import Path

from .chunker import extract_audio

TRANSCRIBE_BIN = os.path.join(os.environ.get("HOME", ""), "ygg", "bin", "transcribe")
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")


def _has_transcribe_script() -> bool:
    return os.path.isfile(TRANSCRIBE_BIN) and os.access(TRANSCRIBE_BIN, os.X_OK)


def transcribe_file(
    audio_path: str,
    model: str | None = None,
    timestamps: bool = True,
) -> str | None:
    """Transcribe an audio/video file to text.

    Args:
        audio_path: Path to audio or video file.
        model: Whisper model name (default: large-v3-turbo).
        timestamps: If True, include timestamps in output.

    Returns:
        Transcribed text, or None on failure.
    """
    model = model or DEFAULT_MODEL

    if _has_transcribe_script():
        cmd = [TRANSCRIBE_BIN, "-m", model]
        if timestamps:
            cmd.append("-t")
        cmd.append(audio_path)
    else:
        # Direct whisper-cli fallback
        whisper = _find_whisper_cli()
        if not whisper:
            return None
        models_dir = os.path.expanduser("~/.local/share/whisper-cpp/models")
        model_path = os.path.join(models_dir, f"ggml-{model}.bin")
        if not os.path.isfile(model_path):
            return None
        cmd = [whisper, "-m", model_path, "-f", audio_path, "-np"]
        if not timestamps:
            cmd.append("--no-timestamps")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ, "PATH": f"{os.environ.get('HOME', '')}/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:{os.environ.get('PATH', '')}"},
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def transcribe_video_chunk(
    chunk_path: str,
    model: str | None = None,
) -> tuple[str | None, str | None]:
    """Transcribe a video chunk. Extracts audio first.

    Returns:
        (plain_text, timestamped_text) -- either may be None on failure.
    """
    # Extract audio to WAV
    try:
        wav_path = extract_audio(chunk_path)
    except Exception:
        return None, None

    try:
        plain = transcribe_file(wav_path, model=model, timestamps=False)
        timestamped = transcribe_file(wav_path, model=model, timestamps=True)
        return plain, timestamped
    finally:
        # Clean up temp WAV
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _find_whisper_cli() -> str | None:
    """Find whisper-cli binary."""
    import shutil
    for name in ["whisper-cli", "whisper-cpp"]:
        path = shutil.which(name)
        if path:
            return path
    for loc in ["/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"]:
        if os.path.isfile(loc):
            return loc
    return None
