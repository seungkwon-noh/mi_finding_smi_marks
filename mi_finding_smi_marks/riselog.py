from __future__ import annotations

import json
import logging
import sys


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "levelname": record.levelname,
            "message": record.getMessage(),
            "asctime": self.formatTime(record, self.datefmt),
            "filename": record.filename,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def getMyLogger(name: str = "mi-finding-smi-marks") -> logging.Logger:
    """Keep the original logger entry point while logging to OpenFaaS stdout."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
