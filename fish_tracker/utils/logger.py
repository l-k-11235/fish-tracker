import logging
import os

_global_log_level = logging.INFO
_log_file_path = None


def resolve_log_level(level):
    """Convertit un niveau en str ou int vers un niveau logging."""
    if isinstance(level, str):
        return logging._nameToLevel.get(level.upper(), logging.INFO)
    return level


def set_global_log_level(level):
    global _global_log_level
    _global_log_level = resolve_log_level(level)


def set_log_file(path: str):
    global _log_file_path
    _log_file_path = path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(_global_log_level)

        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if _log_file_path:
            os.makedirs(os.path.dirname(_log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(_log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False  # avoid duplicate logs
    return logger
