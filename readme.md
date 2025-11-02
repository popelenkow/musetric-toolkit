# Musetric Toolkit

Standalone CLI tool extracted from [Musetric](https://github.com/popelenkow/Musetric) so it can be installed from GitHub releases and run worker scripts directly from the terminal.

## Installation

Install the package directly from the latest GitHub release, then download the default BSRoformer checkpoints:
```bash
uv tool install --python 3.13.2 https://github.com/popelenkow/musetric-toolkit/releases/download/v0.0.1/musetric_toolkit-0.0.1-1cpu-py3-none-any.whl
# or cuda
# uv tool install --python 3.13.2 https://github.com/popelenkow/musetric-toolkit/releases/download/v0.0.1/musetric_toolkit-0.0.1-1cuda-py3-none-any.whl
musetric-download-models
```

For local development, install the CLI in editable mode with [`uv`](https://github.com/astral-sh/uv), then download the BSRoformer checkpoints and configs:
```bash
uv tool install --python 3.13.2 --editable ".[cpu]"
# or cuda
# uv tool install --python 3.13.2 --editable ".[cuda]"
musetric-download-models
```

## CLI Usage

```bash
musetric-separate-audio \
  --source-path /path/to/input.wav \  # input audio file
  --vocal-path /path/to/output-vocals.wav \  # output path for the vocal track
  --instrumental-path /path/to/output-instrumental.wav \  # output path for the instrumental track
  --sample-rate 44100 \  # target sample rate (e.g. 44100)
  --output-format wav  # export format (e.g. wav)
```

## Dependencies

### BSRoformer Neural Network

- **Source:** https://github.com/lucidrains/BS-RoFormer by Phil Wang (MIT)
- **Usage:** Audio source separation model (adapted)

### Research & Development Support

- **Thanks to:** https://github.com/nomadkaraoke/python-audio-separator (MIT)
- **Usage:** Research tool that helped validate BSRoformer approach and integration patterns

## License

Musetric Toolkit is [MIT licensed](https://github.com/popelenkow/Musetric/blob/main/license.md).
