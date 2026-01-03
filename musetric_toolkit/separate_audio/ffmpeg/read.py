import numpy as np

from musetric_toolkit.separate_audio.ffmpeg.runner import run_ffmpeg


def read_audio_file(
    audio_file_path: str,
    sample_rate: int,
    channels: int,
) -> np.ndarray:
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        audio_file_path,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]

    raw_bytes = run_ffmpeg(
        ffmpeg_command,
        capture_stdout=True,
        context="ffmpeg failed to read audio",
    )
    if not raw_bytes:
        raise RuntimeError("ffmpeg produced no audio data")

    samples = np.frombuffer(raw_bytes, dtype=np.float32)
    if samples.size % channels != 0:
        num_frames = samples.size // channels
        samples = samples[: num_frames * channels]
    num_frames = samples.size // channels
    audio_channels_first = samples.reshape(num_frames, channels).T

    return np.ascontiguousarray(audio_channels_first, dtype=np.float32)
