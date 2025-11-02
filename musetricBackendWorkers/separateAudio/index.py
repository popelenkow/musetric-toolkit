import argparse
import logging
import sys

from musetricBackendWorkers.common import envs
from musetricBackendWorkers.common.logger import setupLogging
from musetricBackendWorkers.common.modelFiles import ensureModelFiles
from musetricBackendWorkers.separateAudio.bsRoformerSeparator import BSRoformerSeparator
from musetricBackendWorkers.separateAudio.systemInfo import (
    ensureFfmpeg,
    printAccelerationInfo,
    setupTorchOptimization,
)


def parseArguments():
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
    args = parseArguments()

    setupLogging(args.log_level)

    try:
        ensureModelFiles(
            envs.model_checkpoint_path,
            envs.model_config_path,
        )
        ensureFfmpeg()
        printAccelerationInfo()
        setupTorchOptimization()

        separator = BSRoformerSeparator(
            modelCheckpointPath=envs.model_checkpoint_path,
            modelConfigPath=envs.model_config_path,
            sampleRate=args.sample_rate,
            outputFormat=args.output_format,
        )
        separator.separateAudio(
            sourcePath=args.source_path,
            vocalPath=args.vocal_path,
            instrumentalPath=args.instrumental_path,
        )
    except Exception as error:
        logging.error("Audio separation failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
