import logging
import os
import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any


def setup_logging(log_level: str = "INFO",
                  log_file: str = "fod_detection.log",
                  log_to_console: bool = True,
                  max_log_size: int = 10485760,  # 10 MB
                  backup_count: int = 5) -> logging.Logger:
    """
    Set up application logging

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_to_console: Whether to log to console
        max_log_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep

    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.dirname(log_file)
    if logs_dir and not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Convert string log level to numeric
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_log_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Create and return application logger
    logger = logging.getLogger("FOD")
    logger.info("Logging initialized at %s", datetime.datetime.now().isoformat())
    logger.info("Log level set to %s", log_level)

    return logger


class LogBuffer:
    """
    Buffer for storing recent log messages
    """

    def __init__(self, max_entries: int = 100):
        """
        Initialize log buffer

        Args:
            max_entries: Maximum number of log entries to store
        """
        self.max_entries = max_entries
        self.entries = []
        self.handler = None

    def install(self):
        """Install the buffer handler in the root logger"""
        if self.handler is not None:
            return

        class BufferHandler(logging.Handler):
            def __init__(self, buffer):
                super().__init__()
                self.buffer = buffer

            def emit(self, record):
                self.buffer.add_entry({
                    "timestamp": datetime.datetime.fromtimestamp(record.created),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name
                })

        self.handler = BufferHandler(self)
        logging.getLogger().addHandler(self.handler)

    def uninstall(self):
        """Remove the buffer handler from the root logger"""
        if self.handler is not None:
            logging.getLogger().removeHandler(self.handler)
            self.handler = None

    def add_entry(self, entry: Dict[str, Any]):
        """
        Add a log entry to the buffer

        Args:
            entry: Log entry dictionary
        """
        self.entries.append(entry)

        # Trim buffer if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_entries(self, count: Optional[int] = None,
                    level: Optional[str] = None) -> list:
        """
        Get recent log entries

        Args:
            count: Maximum number of entries to return
            level: Filter by log level

        Returns:
            List of log entries
        """
        # Apply level filter if specified
        if level:
            filtered = [e for e in self.entries if e["level"] == level.upper()]
        else:
            filtered = self.entries

        # Apply count limit
        if count is not None:
            return filtered[-count:]
        else:
            return filtered