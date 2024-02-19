import sys

from loguru import logger

logger.remove(0)
logger.add(
    sys.stderr,
    format="<cyan>{time:DD-MM-YYYY HH:mm:ss}</cyan> | <lvl>{level}</lvl> | {message}",
)
