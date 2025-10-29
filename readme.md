# Musetric Toolkit

Standalone CLI tool extracted from [Musetric](https://github.com/popelenkow/Musetric) so it can be installed from GitHub releases and run worker scripts directly from the terminal.

## Installation

Install the package directly from the latest GitHub release (Python 3.13.2):\
`pip install https://github.com/popelenkow/musetric-toolkit/releases/latest/download/musetric_toolkit-0.1.0-py3-none-any.whl`

## CLI Usage

```bash
musetric-separate-audio \
  --source-path /path/to/input.wav \
  --vocal-path /path/to/output-vocals.wav \
  --instrumental-path /path/to/output-instrumental.wav \
  --model-path /path/to/model.ckpt \
  --config-path /path/to/config.yaml \
  --sample-rate 44100 \
  --output-format wav
```

- `--source-path` — input audio file.
- `--vocal-path` — output path for the vocal track.
- `--instrumental-path` — output path for the instrumental track.
- `--model-path` — BSRoformer model checkpoint.
- `--config-path` — BSRoformer config file.
- `--sample-rate` — target sample rate (e.g. `44100`).
- `--output-format` — export format (e.g. `wav`).

## Dependencies

- **BSRoformer Neural Network:** https://github.com/lucidrains/BS-RoFormer (MIT) — audio source separation model.

## License

Musetric Toolkit is [MIT licensed](https://github.com/popelenkow/Musetric/blob/main/license.md).
