"""Centralized logging for VoiceBot."""
import logging, sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.settings import settings

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); logger.addHandler(ch)
    log_path = Path(settings.LOGS_DIR) / (log_file or "voicebot.log")
    fh = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    logger.propagate = False
    return logger
