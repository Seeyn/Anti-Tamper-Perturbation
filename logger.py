import logging
import os

# Suppress TensorFlow and other library noises before they even load
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback for better error visualization
install(show_locals=True)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(level="INFO"):
    # Remove default handler
    logger.remove()

    # Add Rich handler
    logger.add(
        RichHandler(
            console=Console(width=150),
            show_path=False,
            enable_link_path=False,
            rich_tracebacks=True,
            markup=True,
        ),
        format="{message}",
        level=level,
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress chatty libraries
    for name in ["huggingface_hub", "transformers", "diffusers", "accelerate", "PIL"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Suppress warnings from libraries
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


setup_logger()

# Export logger
__all__ = ["logger", "setup_logger"]
