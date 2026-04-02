import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

logging_str = "[%(asctime)s] %(levelname)s: %(message)s"

file_handler = TimedRotatingFileHandler(
    LOG_FILE_PATH,
    when="midnight",
    backupCount=7,
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(logging_str))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(logging_str))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

logger = logging.getLogger("Amazon_AI_Search")
