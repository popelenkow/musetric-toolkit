import json
import logging
import re
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    levelMap = {"DEBUG": "debug", "INFO": "info", "WARNING": "warn", "ERROR": "error"}

    def format(self, record):
        mappedLevel = self.levelMap.get(record.levelname, "info")
        logEntry = {"level": mappedLevel, "message": record.getMessage()}
        return json.dumps(logEntry)


class StreamToLogger:
    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        stream,
        suppress_patterns: list[re.Pattern[str]] | None = None,
    ) -> None:
        self._logger = logger
        self._level = level
        self._stream = stream
        self._buffer = ""
        self._suppress_patterns = suppress_patterns or []
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _should_suppress(self, line: str) -> bool:
        for pattern in self._suppress_patterns:
            if pattern.search(line):
                return True
        return False

    def write(self, message: str | bytes) -> int:
        if not message:
            return 0
        if isinstance(message, bytes):
            message = message.decode(self.encoding, errors=self.errors)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line and not self._should_suppress(line):
                self._logger.log(self._level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer.rstrip("\r")
            if line and not self._should_suppress(line):
                self._logger.log(self._level, line)
            self._buffer = ""
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        try:
            return self._stream.fileno()
        except Exception:
            return -1

    def writable(self) -> bool:
        return True


def setupLogging(level: str):
    levelMap = {"debug": "DEBUG", "info": "INFO", "warn": "WARNING", "error": "ERROR"}

    handler = logging.StreamHandler(sys.__stderr__)
    handler.setFormatter(JSONFormatter())

    rootLogger = logging.getLogger()
    pythonLevel = levelMap.get(level, "INFO")
    numericLevel = getattr(logging, pythonLevel)
    rootLogger.setLevel(numericLevel)
    handler.setLevel(numericLevel)

    for existingHandler in rootLogger.handlers[:]:
        rootLogger.removeHandler(existingHandler)

    rootLogger.addHandler(handler)

    logging.captureWarnings(True)


def redirectStdStreams(
    stdoutLevel: str = "info",
    stderrLevel: str = "warn",
    suppress_patterns: list[re.Pattern[str]] | None = None,
) -> None:
    levelMap = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
    }
    if not isinstance(sys.stdout, StreamToLogger):
        sys.stdout = StreamToLogger(
            logging.getLogger("stdout"),
            levelMap.get(stdoutLevel, logging.INFO),
            sys.__stdout__,
            suppress_patterns=suppress_patterns,
        )
    if not isinstance(sys.stderr, StreamToLogger):
        sys.stderr = StreamToLogger(
            logging.getLogger("stderr"),
            levelMap.get(stderrLevel, logging.WARNING),
            sys.__stderr__,
            suppress_patterns=suppress_patterns,
        )


def sendMessage(message: dict[str, Any]) -> None:
    print(json.dumps(message), flush=True)
