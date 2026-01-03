import argparse
import logging
import sys

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import setup_logging
from musetric_toolkit.common.model_files import ensure_model_files
from musetric_toolkit.separate_audio.bs_roformer_separator import BSRoformerSeparator
from musetric_toolkit.separate_audio.system_info import (
    ensure_ffmpeg,
    print_acceleration_info,
    setup_torch_optimization,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Separate audio into vocal and instrumental parts"
    )
    parser.add_argument(
        "--source-path", required=True, help="Path to source audio file"
    )
    parser.add_argument(
        "--vocal-path", required=True, help="Path for vocal output file"
    )
    parser.add_argument(
        "--instrumental-path",
        required=True,
        help="Path for instrumental output file",
    )
    parser.add_argument(
        "--sample-rate",
        required=True,
        type=int,
        help="Sample rate for separation",
    )
    parser.add_argument(
        "--output-format",
        required=True,
        help="Audio format for separated stems",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    setup_logging(args.log_level)

    try:
        ensure_model_files(
            envs.model_checkpoint_path,
            envs.model_config_path,
        )
        ensure_ffmpeg()
        print_acceleration_info()
        setup_torch_optimization()

        separator = BSRoformerSeparator(
            model_checkpoint_path=envs.model_checkpoint_path,
            model_config_path=envs.model_config_path,
            sample_rate=args.sample_rate,
            output_format=args.output_format,
        )
        separator.separate_audio(
            source_path=args.source_path,
            vocal_path=args.vocal_path,
            instrumental_path=args.instrumental_path,
        )
    except Exception as error:
        logging.error("Audio separation failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
