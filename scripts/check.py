import argparse
import shlex
import subprocess
import sys


def run_commands(commands):
    for cmd in commands:
        cmd_args = shlex.split(cmd) if isinstance(cmd, str) else cmd
        result = subprocess.run(cmd_args, check=False)

        if result.returncode != 0:
            sys.exit(1)


def check():
    commands = [
        "uv run ruff check . --no-cache",
        "uv run isort . --check-only",
        "uv run black . --check",
    ]
    run_commands(commands)


def fix():
    commands = [
        ["ruff", "check", ".", "--fix", "--no-cache"],
        ["isort", "."],
        ["black", "."],
    ]
    run_commands(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linting checks")
    parser.add_argument("--fix", action="store_true", help="Fix linting issues")
    args = parser.parse_args()

    if args.fix:
        fix()
    else:
        check()
