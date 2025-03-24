import logging
import requests
import time
import os
from typing import Optional, Union, Dict, Any

from notifications.base import BaseNotifier

logger = logging.getLogger("FOD.Notifications.Telegram")


class TelegramNotifier(BaseNotifier):
    """
    Send notifications via Telegram bot
    """

    def __init__(self, bot_token: str, chat_id: str,
                 thread_id: Optional[int] = None,
                 min_severity: int = 1,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """
        Initialize the Telegram notifier

        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
            thread_id: Optional message thread ID for forum/topic groups
            min_severity: Minimum severity level to trigger notifications (1=low, 2=medium, 3=high)
            max_retries: Maximum number of retries for failed messages
            retry_delay: Delay between retries in seconds
        """
        super().__init__(name="Telegram", min_severity=min_severity)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Validate bot token and chat ID
        if not bot_token or not chat_id:
            logger.warning("Telegram notifier initialized with empty bot token or chat ID")
        else:
            logger.info(f"Telegram notifier initialized with chat_id: {chat_id}, thread_id: {thread_id}")

    def is_configured(self) -> bool:
        """Check if the notifier is properly configured"""
        return bool(self.bot_token) and bool(self.chat_id)

    def _check_internet_connection(self) -> bool:
        """
        Check if internet connection is available

        Returns:
            True if connection is available, False otherwise
        """
        try:
            requests.get("https://api.telegram.org", timeout=5)
            return True
        except requests.RequestException:
            return False

    def send(self, alert: Any) -> bool:
        """
        Send alert notification via Telegram

        Args:
            alert: The alert to send

        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            logger.error("Telegram notifier not properly configured")
            return False

        if alert.severity < self.min_severity:
            logger.debug(f"Alert severity {alert.severity} below minimum {self.min_severity}, skipping")
            return True  # Skip but return success

        # Print detailed information about what we're sending
        logger.info(
            f"Sending Telegram notification for ROI {alert.roi_name} with {sum(alert.class_counts.values())} objects")

        # Check internet connection
        if not self._check_internet_connection():
            logger.error("No internet connection available for Telegram notification")
            return False

        # First send text message
        message_success = self._send_text_message(alert)

        # Then send image if available
        image_success = True
        if message_success and alert.snapshot_path and os.path.exists(alert.snapshot_path):
            image_success = self._send_image(alert)

        return message_success and image_success

    def _send_text_message(self, alert: Any) -> bool:
        """
        Send text message via Telegram

        Args:
            alert: The alert to send

        Returns:
            True if successful, False otherwise
        """
        # Format the message
        message = self.format_message(alert)

        # Build API URL and parameters
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        # Add thread ID if provided
        if self.thread_id and self.thread_id > 0:
            payload["message_thread_id"] = self.thread_id

        # Send with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending Telegram message attempt {attempt + 1}/{self.max_retries}")
                response = requests.post(url, data=payload, timeout=10)

                if response.status_code == 200:
                    logger.info(f"Successfully sent Telegram message for alert")
                    return True
                else:
                    try:
                        error_info = response.json() if response.text else "Unknown error"
                        logger.error(f"Telegram API error (attempt {attempt + 1}/{self.max_retries}): {error_info}")
                    except:
                        logger.error(
                            f"Telegram API error (attempt {attempt + 1}/{self.max_retries}): Status code {response.status_code}")
            except Exception as e:
                logger.error(f"Error sending Telegram message (attempt {attempt + 1}/{self.max_retries}): {e}")

            # Delay before retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return False

    def _send_image(self, alert: Any) -> bool:
        """
        Send image via Telegram

        Args:
            alert: The alert with snapshot to send

        Returns:
            True if successful, False otherwise
        """
        if not alert.snapshot_path or not os.path.exists(alert.snapshot_path):
            logger.error(f"Cannot send image: snapshot path invalid or file not found - {alert.snapshot_path}")
            return False

        # Format caption
        caption = self.format_message(alert)

        # Build API URL
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

        # Prepare payload
        payload = {
            "chat_id": self.chat_id,
            "caption": caption,
            "parse_mode": "HTML"
        }

        # Add thread ID if provided
        if self.thread_id and self.thread_id > 0:
            payload["message_thread_id"] = self.thread_id

        # Open image file
        try:
            with open(alert.snapshot_path, "rb") as photo:
                files = {"photo": photo}

                # Send with retries
                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"Sending Telegram photo attempt {attempt + 1}/{self.max_retries}")
                        response = requests.post(url, data=payload, files=files, timeout=30)

                        if response.status_code == 200:
                            logger.info(f"Successfully sent Telegram photo for alert")
                            return True
                        else:
                            try:
                                error_info = response.json() if response.text else "Unknown error"
                                logger.error(
                                    f"Telegram API error (attempt {attempt + 1}/{self.max_retries}): {error_info}")
                            except:
                                logger.error(
                                    f"Telegram API error (attempt {attempt + 1}/{self.max_retries}): Status code {response.status_code}")
                    except Exception as e:
                        logger.error(f"Error sending Telegram photo (attempt {attempt + 1}/{self.max_retries}): {e}")

                    # Delay before retry
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
        except Exception as e:
            logger.error(f"Error opening snapshot file {alert.snapshot_path}: {e}")

        return False