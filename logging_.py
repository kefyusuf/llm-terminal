import sys
from pathlib import Path

from loguru import logger

import config


def setup_logging():
    logger.remove()

    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    log_dir = config.settings.cache_db_path.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    return logger


def get_logger(name: str):
    return logger.bind(name=name)
