import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable

logger = logging.getLogger("FOD.Config")


class ConfigManager:
    """
    Manager for application configuration with component synchronization
    """

    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the configuration manager

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = {}
        # In utils/config.py, update defaults in the ConfigManager.__init__ method
        self.defaults = {
            # Camera settings
            "rtsp_url": "rtsp://admin:password@192.168.102.15:554/Stream/Channels/101?resolution=1920x1080",
            "resize_width": 1920,
            "resize_height": 1080,
            "auto_connect_camera": True,  # Changed to True for convenience

            # RTSP advanced settings - key latency optimizations
            "rtsp_transport": "udp",  # CHANGED from tcp to udp for lower latency
            "rtsp_auto_switch": True,
            "buffer_size": 10,  # REDUCED from 30 to 10 for lower latency
            "enable_adaptive_buffer": True,
            "connection_timeout": 10.0,  # REDUCED from 15.0
            "frame_timeout": 3.0,  # REDUCED from 5.0
            "enable_frame_interpolation": False,  # DISABLED - can add latency
            "adaptive_quality": True,

            # New optimization settings
            "enable_frame_skip": True,
            "enable_half_precision": True,
            "detection_input_size": [640, 640],  # Model input size
            "max_detection_frequency": 15,  # Hz (process up to 15 detections per second)

            # Alert settings
            "object_threshold": 1,
            "alert_cooldown": 20,

            # Sound alerts
            "enable_sound_alert": False,
            "sound_alert_file": "alert_sound.mp3",

            # YOLO settings
            "yolo_model_path": "yolo11n.pt",
            "use_gpu": True,
            "yolo_confidence_threshold": 0.25,
            "classes_of_interest": list(range(40)),

            # Model transition settings (new)
            "enable_auto_class_mapping": True,  # Automatically map classes on model transition
            "prompt_for_class_mapping": True,  # Show prompt for manual class mappings
            "preserve_custom_classes": True,  # Keep custom class definitions on model change
            "synchronize_components": True,  # Automatically sync ROI, detector, etc.

            # Class management settings (new)
            "class_priority_inheritance": True,  # Inherit priorities when mapping classes
            "custom_class_id_start": 100,  # Starting ID for custom classes

            # Telegram notifications
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "telegram_message_thread_id": 0,

            # Email notifications
            "email_enabled": False,
            "email_smtp_server": "smtp.gmail.com",
            "email_smtp_port": 587,
            "email_use_ssl": True,
            "email_username": "",
            "email_password": "",
            "email_from": "",
            "email_to": "",

            # Logging
            "log_level": "INFO"
        }

        # Change listeners
        self._listeners = []

        # Load configuration
        self.load()

    def add_listener(self, listener: Callable[[str, Any], None]):
        """
        Add a listener for configuration changes

        Args:
            listener: Function that accepts (key, value) for config changes
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable):
        """
        Remove a configuration change listener

        Args:
            listener: Listener to remove
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self, key: str, value: Any):
        """
        Notify listeners of configuration change

        Args:
            key: Configuration key that changed
            value: New value
        """
        for listener in self._listeners:
            try:
                listener(key, value)
            except Exception as e:
                logger.error(f"Error notifying config listener for '{key}': {e}")

    def load(self) -> bool:
        """
        Load configuration from file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = yaml.safe_load(f)

                    if loaded_config is None:
                        loaded_config = {}

                    # Update config with loaded values
                    self.config.update(loaded_config)

                    logger.info(f"Configuration loaded from {self.config_file}")
                    return True
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                self.config = self.defaults.copy()
                return False
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self.defaults.copy()
            return False

    def save(self) -> bool:
        """
        Save configuration to file

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, self.defaults.get(key, default))

    def set(self, key: str, value: Any, notify: bool = True):
        """
        Set a configuration value

        Args:
            key: Configuration key
            value: Configuration value
            notify: Whether to notify listeners
        """
        # Check if the value is actually changing
        current_value = self.get(key)
        if current_value == value:
            return

        # Update the value
        self.config[key] = value

        # Notify listeners if requested
        if notify:
            self._notify_listeners(key, value)

        # Special handling for certain settings
        self._handle_special_settings(key, value)

    def _handle_special_settings(self, key: str, value: Any):
        """
        Special handling for certain settings that affect multiple components

        Args:
            key: Setting key
            value: Setting value
        """
        # Handle YOLO model path change
        if key == "yolo_model_path" and self.get("synchronize_components", True):
            logger.info(f"YOLO model changed to {value}. Synchronizing components...")
            # This would be handled by the ModelTransitionManager in main_window.py

        # Handle class priority changes
        elif key.startswith("class_priority_") and self.get("synchronize_components", True):
            class_id = int(key.split("_")[-1])
            logger.info(f"Class {class_id} priority changed to {value}. Updating alert calculations...")
            # This would update the ClassManager through a listener

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values

        Returns:
            Dictionary with all configuration values
        """
        # Start with defaults
        result = self.defaults.copy()

        # Override with current config
        result.update(self.config)

        return result

    def reset_to_defaults(self):
        """Reset all configuration to defaults"""
        old_config = self.config.copy()
        self.config = self.defaults.copy()

        # Notify about all changed settings
        for key, value in self.config.items():
            if key not in old_config or old_config[key] != value:
                self._notify_listeners(key, value)

        logger.info("Configuration reset to defaults")

    def export_to_file(self, file_path: str) -> bool:
        """
        Export configuration to a different file

        Args:
            file_path: Path to export configuration to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Choose format based on file extension
            if file_path.lower().endswith('.json'):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.get_all(), f, indent=4)
            else:
                # Default to YAML
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(self.get_all(), f, default_flow_style=False, sort_keys=False)

            logger.info(f"Configuration exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False

    def import_from_file(self, file_path: str) -> bool:
        """
        Import configuration from a file

        Args:
            file_path: Path to import configuration from

        Returns:
            True if successful, False otherwise
        """
        try:
            # Choose format based on file extension
            if file_path.lower().endswith('.json'):
                with open(file_path, "r", encoding="utf-8") as f:
                    imported_config = json.load(f)
            else:
                # Default to YAML
                with open(file_path, "r", encoding="utf-8") as f:
                    imported_config = yaml.safe_load(f)

            if not isinstance(imported_config, dict):
                logger.error(f"Invalid configuration format in {file_path}")
                return False

            # Store old config for change detection
            old_config = self.config.copy()

            # Update config with imported values
            self.config.update(imported_config)

            # Notify about changed settings
            for key, value in self.config.items():
                if key not in old_config or old_config[key] != value:
                    self._notify_listeners(key, value)

            logger.info(f"Configuration imported from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False