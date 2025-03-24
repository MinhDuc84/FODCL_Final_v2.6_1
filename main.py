#!/usr/bin/env python3
"""
FOD Detection System - Main Entry Point

A modern application for Foreign Object Detection using YOLOv8
with an improved PyQt5 GUI interface.
"""

import sys
import os
import argparse
import logging
import time
import traceback

from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

# Import utilities
from utils.logging import setup_logging
from utils.config import ConfigManager

# Import UI
from ui.main_window import MainWindow


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FOD Detection System")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/fod_detection.log",
        help="Path to log file"
    )

    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Disable splash screen"
    )

    return parser.parse_args()


def setup_exception_handling(logger):
    """Set up global exception handler"""

    def exception_handler(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        # Show error dialog
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        QMessageBox.critical(
            None,
            "Application Error",
            f"An unhandled error occurred:\n\n{error_msg}\n\nThe application will now close."
        )

        # Exit the application
        sys.exit(1)

    # Install exception handler
    sys.excepthook = exception_handler


def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_arguments()

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    # Set up logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        log_to_console=True
    )

    # Set up exception handling
    setup_exception_handling(logger)

    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("FOD Detection System")
    app.setApplicationVersion("2.0")
    app.setStyle("Fusion")  # Modern style

    # Set application icon
    # app.setWindowIcon(QIcon("ui/resources/icon.png"))

    # Show splash screen
    splash = None
    if not args.no_splash:
        try:
            # splash_pixmap = QPixmap("ui/resources/splash.png")
            # Use a generated splash screen if image not available
            splash_pixmap = QPixmap(400, 300)
            splash_pixmap.fill(Qt.white)

            splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
            splash.showMessage(
                "Starting FOD Detection System...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.black
            )
            splash.show()
            app.processEvents()
        except Exception as e:
            logger.warning(f"Failed to show splash screen: {e}")

    # Load configuration
    logger.info("Loading configuration from %s", args.config)
    config = ConfigManager(args.config)

    # Initialize main window with a delay to show splash screen
    main_window = None

    def init_main_window():
        nonlocal main_window
        try:
            # Create main window
            main_window = MainWindow()

            # Hide splash screen if exists
            if splash:
                splash.finish(main_window)

            # Show main window
            main_window.show()
            logger.info("Application started successfully")

        except Exception as e:
            logger.critical("Failed to initialize main window", exc_info=True)
            if splash:
                splash.hide()
            QMessageBox.critical(
                None,
                "Initialization Error",
                f"Failed to start the application: {e}"
            )
            sys.exit(1)

    # Use timer to show splash for at least 1 second
    if splash:
        QTimer.singleShot(1000, init_main_window)
    else:
        init_main_window()

    # Start the event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())