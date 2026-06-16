import logging
from logging.config import dictConfig


def configure_logging(debug: bool = False) -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "DEBUG" if debug else "INFO",
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
