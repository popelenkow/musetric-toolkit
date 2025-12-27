# Musetric Toolkit

Standalone CLI tool extracted from [Musetric](https://github.com/popelenkow/musetric) so it can be installed from GitHub releases and run worker scripts directly from the terminal.

## Installation

Install the package directly from the latest GitHub release, then download the default BSRoformer checkpoints:
```bash
uv tool install --python 3.13.2 \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cpu \  # --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  https://github.com/popelenkow/musetric-toolkit/releases/download/v0.0.4/musetric_toolkit-0.0.4-py3-none-any.whl

musetric-download-models
```

For local development, install the CLI in editable mode with [`uv`](https://github.com/astral-sh/uv), then download the BSRoformer checkpoints and configs:
```bash
uv tool install --python 3.13.2 --editable . \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cpu \  # --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

musetric-download-models
```

## CLI Usage

```bash
musetric-separate \
  --source-path /path/to/input.wav \  # input audio file
  --vocal-path /path/to/output-vocals.wav \  # output path for the vocal track
  --instrumental-path /path/to/output-instrumental.wav \  # output path for the instrumental track
  --sample-rate 44100 \  # target sample rate (e.g. 44100)
  --output-format wav  # export format (e.g. wav)
```

```bash
musetric-transcribe \
  --audio-path /path/to/vocals.wav \  # input vocal audio file
  --result-path /path/to/transcription.json \  # output JSON file
```

## Dependencies

### BSRoformer Neural Network

- **Source:** https://github.com/lucidrains/BS-RoFormer by Phil Wang (MIT)
- **Usage:** Audio source separation model (adapted)
- **Thanks to:** https://github.com/nomadkaraoke/python-audio-separator (MIT) — research tool that helped validate the BSRoformer approach and integration patterns

### WhisperX Speech Transcription

- **Source:** https://github.com/m-bain/whisperX by Max Bain (MIT)
- **Usage:** Speech-to-text + word-level alignment for `musetric-transcribe`

## License

Musetric Toolkit is [MIT licensed](https://github.com/popelenkow/Musetric/blob/main/license.md).
