import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union

# Import the class manager
from storage.class_manager import ClassManager

# Import Alert class but handle circular import
try:
    from core.alert_manager import Alert
except ImportError:
    Alert = Any  # Type stub

logger = logging.getLogger("FOD.Notifications")


class BaseNotifier(ABC):
    """
    Base class for notification channels
    """

    # Class manager instance
    _class_manager = None

    def __init__(self, name: str = "Generic Notifier", min_severity: int = 1):
        """
        Initialize the notifier

        Args:
            name: Name of the notifier
            min_severity: Minimum severity level to trigger notifications (1=low, 2=medium, 3=high)
        """
        self.name = name
        self.min_severity = min_severity
        logger.debug(f"Initialized {self.name}")

    @classmethod
    def get_class_manager(cls) -> ClassManager:
        """Get class manager instance, creating it if needed"""
        if cls._class_manager is None:
            cls._class_manager = ClassManager()
        return cls._class_manager

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """
        Send a notification for an alert

        Args:
            alert: The alert to send notification for

        Returns:
            True if successful, False otherwise
        """
        pass

    def format_message(self, alert: Alert) -> str:
        """
        Format alert into a text message

        Args:
            alert: The alert to format

        Returns:
            Formatted message
        """
        severity_text = {
            1: "LOW",
            2: "MEDIUM",
            3: "HIGH"
        }.get(alert.severity, "UNKNOWN")

        # Format timestamp
        timestamp = alert.datetime.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"[ALERT] {severity_text} - ROI: {alert.roi_name}",
            f"Camera: {alert.camera_id}",
            f"Time: {timestamp}",
            "Detections:"
        ]

        # Get class names dictionary from class manager
        class_names = self.get_class_manager().get_class_names()

        # Get priority levels
        priority_levels = {
            1: "Low",
            2: "Medium",
            3: "High",
            4: "Critical"
        }

        # Add detection counts with priority levels
        total_objects = 0
        for class_id, count in alert.class_counts.items():
            if count > 0:
                class_id = int(class_id)
                class_name = class_names.get(class_id, f"Unknown-{class_id}")
                priority = alert.class_priorities.get(class_id, 1)
                priority_text = priority_levels.get(priority, "Unknown")
                lines.append(f"- {class_name}: {count} [Priority: {priority_text}]")
                total_objects += count

        # Add total count
        lines.append(f"Total objects: {total_objects}")

        return "\n".join(lines)