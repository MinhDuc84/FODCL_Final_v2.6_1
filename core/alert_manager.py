import time
import logging
import datetime
import os
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable

from storage.database import AlertDatabase
from notifications.base import BaseNotifier
from storage.class_manager import ClassManager  # Add class manager import

logger = logging.getLogger("FOD.AlertManager")


# Modify the Alert class to better integrate with ClassManager

class Alert:
    """
    Represents a detection alert
    """
    # Severity levels
    SEVERITY_LOW = 1
    SEVERITY_MEDIUM = 2
    SEVERITY_HIGH = 3

    # Priority levels for classes (to be used in configuration)
    PRIORITY_LOW = 1
    PRIORITY_MEDIUM = 2
    PRIORITY_HIGH = 3
    PRIORITY_CRITICAL = 4

    # Keep DEFAULT_CLASS_PRIORITIES for backward compatibility
    # Will be used if class_manager fails or is not available
    DEFAULT_CLASS_PRIORITIES = {
        # High priority objects (metal objects, tools that can cause damage)
        0: PRIORITY_HIGH,  # AdjustableClamp
        1: PRIORITY_HIGH,  # AdjustableWrench
        9: PRIORITY_HIGH,  # Hammer
        14: PRIORITY_HIGH,  # MetalPart
        15: PRIORITY_HIGH,  # MetalSheet
        21: PRIORITY_HIGH,  # Pliers
        30: PRIORITY_HIGH,  # Wrench
        31: PRIORITY_HIGH,  # Copper
        32: PRIORITY_HIGH,  # Metallic shine

        # Critical priority objects (extremely dangerous)
        3: PRIORITY_CRITICAL,  # Bolt
        4: PRIORITY_CRITICAL,  # BoltNutSet
        5: PRIORITY_CRITICAL,  # BoltWasher
        16: PRIORITY_CRITICAL,  # Nail
        17: PRIORITY_CRITICAL,  # Nut
        23: PRIORITY_CRITICAL,  # Screw
        27: PRIORITY_CRITICAL,  # Washer

        # Medium priority objects
        2: PRIORITY_MEDIUM,  # Battery
        6: PRIORITY_MEDIUM,  # ClampPart
        7: PRIORITY_MEDIUM,  # Cutter
        8: PRIORITY_MEDIUM,  # FuelCap
        10: PRIORITY_MEDIUM,  # Hose
        12: PRIORITY_MEDIUM,  # LuggagePart
        24: PRIORITY_MEDIUM,  # Screwdriver
        25: PRIORITY_MEDIUM,  # SodaCan
        28: PRIORITY_MEDIUM,  # Wire

        # Low priority objects
        11: PRIORITY_LOW,  # Label
        13: PRIORITY_LOW,  # LuggageTag
        18: PRIORITY_LOW,  # PaintChip
        19: PRIORITY_LOW,  # Pen
        20: PRIORITY_LOW,  # PlasticPart
        22: PRIORITY_LOW,  # Rock
        26: PRIORITY_LOW,  # Tape
        29: PRIORITY_LOW,  # Wood
        33: PRIORITY_LOW,  # Eyebolt
        34: PRIORITY_LOW,  # AsphaltCrack
        35: PRIORITY_LOW,  # FaucetHandle
        36: PRIORITY_LOW,  # Tie-Wrap
        37: PRIORITY_LOW,  # Pitot cover
        38: PRIORITY_LOW,  # Scissors
        39: PRIORITY_LOW,  # NutShell
    }

    # Class manager instance for getting class priorities
    _class_manager = None

    def __init__(self, roi_id: int, roi_name: str,
                 class_counts: Dict[int, int],
                 camera_id: str,
                 timestamp: Optional[float] = None,
                 detections: Optional[List[Dict[str, Any]]] = None,
                 snapshot_path: Optional[str] = None,
                 video_path: Optional[str] = None,
                 severity: int = SEVERITY_MEDIUM,
                 class_priorities: Optional[Dict[int, int]] = None):
        """
        Initialize an alert

        Args:
            roi_id: ID of the ROI that triggered the alert
            roi_name: Name of the ROI
            class_counts: Dictionary mapping class IDs to counts
            camera_id: ID of the camera
            timestamp: Alert timestamp (defaults to current time)
            detections: List of detections that triggered the alert
            snapshot_path: Path to the saved snapshot image
            video_path: Path to the saved video clip
            severity: Alert severity level
            class_priorities: Dictionary mapping class IDs to priority levels
        """
        self.roi_id = roi_id
        self.roi_name = roi_name
        self.class_counts = class_counts
        self.camera_id = camera_id
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp)
        self.detections = detections or []
        self.snapshot_path = snapshot_path
        self.video_path = video_path

        # Initialize class priorities
        if class_priorities is not None:
            self.class_priorities = class_priorities
        else:
            # Get priorities from class manager
            self.class_priorities = self.get_class_priorities()

        # Calculate severity based on priority if not explicitly provided
        if severity == self.SEVERITY_MEDIUM:  # Only recalculate if using default
            self.severity = self._calculate_severity()
        else:
            self.severity = severity

    def to_dict(self):
        """Convert alert to dictionary for storage"""
        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "class_counts": self.class_counts,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "snapshot_path": self.snapshot_path,
            "video_path": self.video_path,
            "severity": self.severity
        }

    @classmethod
    def get_class_manager(cls) -> 'ClassManager':
        """Get the class manager instance, creating it if needed"""
        if cls._class_manager is None:
            # Import here to avoid circular imports
            from storage.class_manager import ClassManager
            cls._class_manager = ClassManager()
        return cls._class_manager

    @classmethod
    def get_class_priorities(cls) -> Dict[int, int]:
        """Get class priorities from the class manager"""
        try:
            # Try to get from class manager first
            priorities = cls.get_class_manager().get_class_priorities()

            # If we got an empty dictionary (which shouldn't happen but just in case),
            # fall back to defaults
            if not priorities:
                logger.warning("Empty priorities from class manager, using defaults")
                return cls.DEFAULT_CLASS_PRIORITIES

            return priorities
        except Exception as e:
            # If anything goes wrong, fall back to default priorities
            logger.error(f"Error getting class priorities from manager: {e}")
            logger.info("Falling back to default class priorities")
            return cls.DEFAULT_CLASS_PRIORITIES

    def _calculate_severity(self) -> int:
        """Calculate alert severity based on detected object priorities"""
        if not self.class_counts:
            return self.SEVERITY_LOW

        highest_priority = self.PRIORITY_LOW
        total_objects = 0

        # Find highest priority object in the alert
        for class_id, count in self.class_counts.items():
            if count > 0:
                priority = self.class_priorities.get(int(class_id), self.PRIORITY_LOW)
                highest_priority = max(highest_priority, priority)
                total_objects += count

        # Map priority levels to severity
        if highest_priority == self.PRIORITY_CRITICAL or total_objects >= 5:
            return self.SEVERITY_HIGH
        elif highest_priority == self.PRIORITY_HIGH or total_objects >= 3:
            return self.SEVERITY_MEDIUM
        else:
            return self.SEVERITY_LOW


class AlertManager:
    """
    Manages alert generation, processing, and notification
    """

    def __init__(self, snapshot_dir: str = "Snapshots",
                 video_dir: str = "EventVideos",
                 db_path: str = "alerts.db",
                 class_manager=None):
        """
        Initialize the alert manager

        Args:
            snapshot_dir: Directory for storing snapshot images
            video_dir: Directory for storing video clips
            db_path: Path to the SQLite database file
            class_manager: ClassManager instance for class lookups
        """
        self.snapshot_dir = snapshot_dir
        self.video_dir = video_dir

        # Create directories if they don't exist
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # Initialize database
        self.db = AlertDatabase(db_path)

        # Notification channels
        self.notifiers: List[BaseNotifier] = []

        # Alert queue and worker thread
        self.alert_queue = queue.Queue()
        self.offline_alerts = []
        self.stop_event = threading.Event()
        self._worker_thread = None

        # Alert history for statistics
        self.alert_events = []  # List of (timestamp, count) tuples

        # Start worker thread immediately
        self.start_worker()

        # Set class manager for Alert class
        if class_manager:
            Alert._class_manager = class_manager
            # Add listener for class changes
            class_manager.add_listener(self._handle_class_change)

        # Load any offline alerts
        self._load_offline_alerts()

        logger.info("Alert Manager initialized with worker thread")

    def _handle_class_change(self, event):
        """
        Handle class change events

        Args:
            event: ClassChangeEvent instance
        """
        # For alert manager, we might want to recompute alert severities
        # if priority definitions change
        if event.action == "update" and "priority" in event.data:
            logger.info(f"Class {event.class_id} priority changed to {event.data['priority']}")
            # Here you could recalculate severities for stored alerts if needed

        # Clear Alert class cache on any class-related changes
        Alert._class_manager = None

    def add_notifier(self, notifier: BaseNotifier):
        """Add a notification channel"""
        self.notifiers.append(notifier)
        logger.info(f"Added notification channel: {notifier.__class__.__name__}")

        # Set class manager for notifier if it has the interface
        if hasattr(notifier, 'set_class_manager') and Alert._class_manager:
            notifier.set_class_manager(Alert._class_manager)

    def start_worker(self):
        """Start the alert processing worker thread"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Alert worker thread is already running")
            return

        self.stop_event.clear()
        self._worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self._worker_thread.start()
        logger.info("Started alert worker thread")

    def _process_alerts(self):
        """Worker thread for processing alerts"""
        logger.info("Alert processing thread started")

        while not self.stop_event.is_set():
            try:
                # First process any offline alerts if possible
                self._process_offline_alerts()

                # Process new alerts from the queue
                try:
                    # Use a short timeout to allow checking stop_event frequently
                    alert = self.alert_queue.get(timeout=1)
                    logger.info(f"Processing alert for ROI {alert.roi_id} ({alert.roi_name})")
                except queue.Empty:
                    continue

                # Store in database
                try:
                    alert_id = self.db.insert_alert(
                        timestamp=alert.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        roi_name=alert.roi_name,
                        roi_index=alert.roi_id,
                        alert_message=json.dumps(alert.class_counts),
                        snapshot_path=alert.snapshot_path or "",
                        video_path=alert.video_path or "",
                        severity=alert.severity,
                        camera_id=alert.camera_id
                    )
                    logger.info(f"Alert stored in database with ID {alert_id}")
                except Exception as e:
                    logger.error(f"Error storing alert in database: {e}")

                # Send notifications
                if self.notifiers:
                    notification_success = self._send_notifications(alert)

                    if not notification_success:
                        # Store for offline processing
                        self.offline_alerts.append(alert)
                        self._save_offline_alerts()
                        logger.warning(f"Alert {alert.roi_id} stored for offline processing")
                else:
                    logger.warning("No notification channels configured. Alert saved to database only.")

                self.alert_queue.task_done()

            except Exception as e:
                logger.error(f"Error in alert processing thread: {e}")
                # Don't let thread crash
                time.sleep(1)

    def _process_offline_alerts(self):
        """Process stored offline alerts"""
        if not self.offline_alerts:
            return

        # Try to send each offline alert
        successfully_sent = []

        for alert in self.offline_alerts:
            if self._send_notifications(alert):
                successfully_sent.append(alert)
                logger.info(f"Successfully sent offline alert for ROI {alert.roi_id}")

        # Remove successfully sent alerts
        for alert in successfully_sent:
            self.offline_alerts.remove(alert)

        if successfully_sent:
            self._save_offline_alerts()

    def _send_notifications(self, alert: Alert) -> bool:
        """
        Send notifications for an alert

        Args:
            alert: The alert to send notifications for

        Returns:
            True if all notifications were sent successfully, False otherwise
        """
        if not self.notifiers:
            logger.warning("No notification channels configured")
            return True

        all_success = True

        for notifier in self.notifiers:
            try:
                # Skip if severity is too low for this notifier
                if alert.severity < notifier.min_severity:
                    logger.debug(
                        f"Skipping {notifier.__class__.__name__} due to low severity {alert.severity} (min: {notifier.min_severity})")
                    continue

                # Send notification
                logger.info(f"Sending notification via {notifier.__class__.__name__}")
                if notifier.send(alert):
                    logger.info(f"Successfully sent notification via {notifier.__class__.__name__}")
                else:
                    logger.error(f"Failed to send notification via {notifier.__class__.__name__}")
                    all_success = False
            except Exception as e:
                logger.error(f"Error sending notification via {notifier.__class__.__name__}: {e}")
                all_success = False

        return all_success

    def _save_offline_alerts(self):
        """Save offline alerts to file"""
        OFFLINE_ALERTS_FILE = "offline_alerts.json"

        try:
            data = [alert.to_dict() for alert in self.offline_alerts]
            with open(OFFLINE_ALERTS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            logger.info(f"Saved {len(self.offline_alerts)} offline alerts to {OFFLINE_ALERTS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save offline alerts: {e}")

    def _load_offline_alerts(self):
        """Load offline alerts from file"""
        OFFLINE_ALERTS_FILE = "offline_alerts.json"

        if not os.path.exists(OFFLINE_ALERTS_FILE):
            return

        try:
            with open(OFFLINE_ALERTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            for alert_data in data:
                # Create alert object from data
                alert = Alert(
                    roi_id=alert_data.get("roi_id"),
                    roi_name=alert_data.get("roi_name"),
                    class_counts=alert_data.get("class_counts", {}),
                    camera_id=alert_data.get("camera_id", "unknown"),
                    timestamp=alert_data.get("timestamp"),
                    snapshot_path=alert_data.get("snapshot_path"),
                    video_path=alert_data.get("video_path"),
                    severity=alert_data.get("severity", Alert.SEVERITY_MEDIUM)
                )
                self.offline_alerts.append(alert)

            logger.info(f"Loaded {len(self.offline_alerts)} offline alerts")
        except Exception as e:
            logger.error(f"Failed to load offline alerts: {e}")

    def stop_worker(self):
        """Stop the alert processing worker thread"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return

        self.stop_event.set()
        self._worker_thread.join(timeout=5)
        logger.info("Stopped alert worker thread")

    def create_alert(self, roi_id: int, roi_name: str,
                     class_counts: Dict[int, int],
                     camera_id: str,
                     frame: Any,
                     save_snapshot: bool = True,
                     start_recording: Optional[Callable] = None) -> Alert:
        """
        Create and queue a new alert

        Args:
            roi_id: ID of the ROI that triggered the alert
            roi_name: Name of the ROI
            class_counts: Dictionary mapping class IDs to counts
            camera_id: ID of the camera
            frame: Current frame with visual information (bounding boxes, labels, etc.)
            save_snapshot: Whether to save a snapshot image
            start_recording: Callback function to start video recording

        Returns:
            The created Alert object
        """
        # Log detailed information about the alert we're creating
        logger.info(f"Creating alert for ROI {roi_id} ({roi_name}) with objects: {class_counts}")

        # Save snapshot if requested - using the processed frame with visual information
        snapshot_path = None
        if save_snapshot:
            # We directly use the frame with visual information
            snapshot_path = self.save_snapshot(frame)
            logger.info(f"Saved alert snapshot to {snapshot_path}")

        # Start recording if callback provided
        video_path = None
        if start_recording:
            try:
                video_path = start_recording(frame)
                logger.info(f"Started alert video recording: {video_path}")
            except Exception as e:
                logger.error(f"Error starting video recording: {e}")

        # Get class priorities from the manager
        class_priorities = Alert.get_class_priorities()

        # Create alert object - severity will be calculated automatically based on detected objects
        alert = Alert(
            roi_id=roi_id,
            roi_name=roi_name,
            class_counts=class_counts,
            camera_id=camera_id,
            snapshot_path=snapshot_path,
            video_path=video_path,
            class_priorities=class_priorities
        )

        # Add to queue for processing
        self.alert_queue.put(alert)
        logger.info(
            f"Alert added to processing queue (Severity: {alert.severity}). Queue size: {self.alert_queue.qsize()}")

        # Record for statistics
        self.alert_events.append((alert.timestamp, sum(class_counts.values())))

        # Return the created alert
        return alert

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = time.time()

        # Count alerts in the last hour and day
        count_hour = sum(count for t, count in self.alert_events if now - t <= 3600)
        count_day = sum(count for t, count in self.alert_events if now - t <= 86400)

        # Get additional stats from database
        db_stats = self.db.get_statistics()

        return {
            "count_last_hour": count_hour,
            "count_last_day": count_day,
            "total_alerts": db_stats.get("total_alerts", 0),
            "alerts_by_severity": db_stats.get("alerts_by_severity", {}),
            "alerts_by_camera": db_stats.get("alerts_by_camera", {}),
            "alerts_by_roi": db_stats.get("alerts_by_roi", {}),
            "alerts_by_day": db_stats.get("alerts_by_day", {})
        }

    def save_snapshot(self, frame) -> str:
        """
        Save a snapshot image

        Args:
            frame: The frame to save (already has visual overlays)

        Returns:
            Path to the saved image
        """
        import cv2

        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}.jpg")

        try:
            # Use the frame directly as it already has overlays
            # Make a copy to avoid modifying the original frame
            annotated_frame = frame.copy()

            # Add timestamp
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated_frame,
                f"Time: {current_time}",
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Save image with annotations
            cv2.imwrite(snapshot_filename, annotated_frame)
            logger.info(f"Snapshot saved to {snapshot_filename}")
            return snapshot_filename
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return ""