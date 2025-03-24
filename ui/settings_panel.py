import logging
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QGroupBox, QFormLayout, QLabel, QLineEdit,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
                             QFileDialog, QComboBox, QSlider, QMessageBox,
                             QTableWidget, QHeaderView, QTableWidgetItem,
                             QScrollArea, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont

from utils.config import ConfigManager

logger = logging.getLogger("FOD.SettingsPanel")


class SettingsPanel(QWidget):
    """
    Widget for configuring application settings
    """

    def __init__(self, config: ConfigManager, class_manager=None, parent=None):
        """
        Initialize the settings panel

        Args:
            config: Configuration manager
            class_manager: Class manager instance for class lookups
            parent: Parent widget
        """
        super().__init__(parent)

        self.config = config
        self.class_manager = class_manager

        # Initialize UI
        self.init_ui()

        # Add listener for class changes if class manager is provided
        if self.class_manager and hasattr(self.class_manager, "add_listener"):
            self.class_manager.add_listener(self._handle_class_change)

        # Load settings
        self.update_settings(config.get_all())

    # Add this utility method to ui/settings_panel.py

    def refresh_class_display(self):
        """
        Force a refresh of the class display
        Call this method when class definitions change or when switching models
        """
        # Update the class checkboxes using the new grid-based layout
        self._update_class_checkboxes()

        # Make sure the classes group is properly sized
        if hasattr(self, 'classes_group'):
            self.classes_group.adjustSize()

        # Update the scroll area if it exists
        for i in range(self.classes_grid_layout.count()):
            item = self.classes_grid_layout.itemAt(i)
            if item and item.widget():
                item.widget().adjustSize()

        # Log the refresh
        logger.info(f"Refreshed class display with {len(self.class_checkboxes)} classes")

    # Update the _handle_class_change method to use the refresh utility
    def _handle_class_change(self, event):
        """
        Handle class change events

        Args:
            event: ClassChangeEvent instance
        """
        # Update class checkboxes when classes change
        if event.action in ["add", "update", "delete", "import", "model_update"]:
            self.refresh_class_display()

            # Log the event
            if event.class_id is not None:
                logger.info(f"Updated display for class {event.class_id} ({event.action})")
            else:
                logger.info(f"Updated display for multiple classes ({event.action})")

    def init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # General settings tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout(camera_group)

        # Display scaling option
        self.display_scale = QComboBox()
        self.display_scale.addItem("100% (Full Resolution)", 1.0)
        self.display_scale.addItem("75%", 0.75)
        self.display_scale.addItem("50%", 0.5)
        self.display_scale.addItem("25%", 0.25)
        camera_layout.addRow("Display Scale:", self.display_scale)

        self.rtsp_url = QLineEdit()
        camera_layout.addRow("RTSP URL:", self.rtsp_url)

        self.resize_width = QSpinBox()
        self.resize_width.setRange(320, 3840)
        self.resize_width.setSingleStep(16)
        camera_layout.addRow("Resize Width:", self.resize_width)

        self.resize_height = QSpinBox()
        self.resize_height.setRange(240, 2160)
        self.resize_height.setSingleStep(16)
        camera_layout.addRow("Resize Height:", self.resize_height)

        # RTSP Transport protocol
        self.rtsp_transport = QComboBox()
        self.rtsp_transport.addItem("TCP (More reliable)", "tcp")
        self.rtsp_transport.addItem("UDP (Lower latency)", "udp")
        camera_layout.addRow("RTSP Transport:", self.rtsp_transport)

        # Auto-switch transport option
        self.rtsp_auto_switch = QCheckBox("Automatically switch transport protocol on connection issues")
        camera_layout.addRow("", self.rtsp_auto_switch)

        # Buffer size
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(5, 100)
        self.buffer_size.setSingleStep(5)
        self.buffer_size.setSuffix(" frames")
        camera_layout.addRow("Buffer Size:", self.buffer_size)

        # Adaptive buffer option
        self.enable_adaptive_buffer = QCheckBox("Dynamically adjust buffer based on network conditions")
        camera_layout.addRow("", self.enable_adaptive_buffer)

        # Auto-connect option
        self.auto_connect = QCheckBox("Auto-connect camera on startup")
        camera_layout.addRow("", self.auto_connect)

        # Camera test button
        self.test_camera_button = QPushButton("Test Connection")
        self.test_camera_button.clicked.connect(self.test_camera_connection)
        camera_layout.addRow("", self.test_camera_button)

        general_layout.addWidget(camera_group)

        # Add new tab for priority configuration
        self._setup_priority_tab()

        # Alert settings
        alert_group = QGroupBox("Alert Settings")
        alert_layout = QFormLayout(alert_group)

        self.object_threshold = QDoubleSpinBox()
        self.object_threshold.setRange(0.1, 100)
        self.object_threshold.setSingleStep(0.1)
        alert_layout.addRow("Object Threshold:", self.object_threshold)

        self.alert_cooldown = QSpinBox()
        self.alert_cooldown.setRange(1, 3600)
        self.alert_cooldown.setSingleStep(5)
        self.alert_cooldown.setSuffix(" seconds")
        alert_layout.addRow("Alert Cooldown:", self.alert_cooldown)

        # Sound alert settings
        self.enable_sound_alert = QCheckBox("Enable Sound Alerts")
        alert_layout.addRow("", self.enable_sound_alert)

        self.sound_alert_file = QLineEdit()
        self.sound_alert_file.setReadOnly(True)

        sound_file_layout = QHBoxLayout()
        sound_file_layout.addWidget(self.sound_alert_file)

        self.browse_sound_button = QPushButton("Browse")
        self.browse_sound_button.clicked.connect(self.browse_sound_file)
        sound_file_layout.addWidget(self.browse_sound_button)

        self.test_sound_button = QPushButton("Test")
        self.test_sound_button.clicked.connect(self.test_sound)
        sound_file_layout.addWidget(self.test_sound_button)

        alert_layout.addRow("Sound File:", sound_file_layout)

        general_layout.addWidget(alert_group)

        # Detection settings tab
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)

        # Model settings
        model_group = QGroupBox("YOLO Model Settings")
        model_layout = QFormLayout(model_group)

        self.yolo_model_path = QLineEdit()
        self.yolo_model_path.setReadOnly(True)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.yolo_model_path)

        self.browse_model_button = QPushButton("Browse")
        self.browse_model_button.clicked.connect(self.browse_model_file)
        model_path_layout.addWidget(self.browse_model_button)

        model_layout.addRow("Model Path:", model_path_layout)

        self.use_gpu = QCheckBox("Use GPU Acceleration")
        model_layout.addRow("", self.use_gpu)

        self.yolo_confidence = QDoubleSpinBox()
        self.yolo_confidence.setRange(0.01, 1.0)
        self.yolo_confidence.setSingleStep(0.05)
        self.yolo_confidence.setDecimals(2)
        model_layout.addRow("Confidence Threshold:", self.yolo_confidence)

        detection_layout.addWidget(model_group)

        # Classes of interest - Create this section only once
        self.classes_group = QGroupBox("Classes of Interest")
        classes_layout = QVBoxLayout(self.classes_group)

        # Create a scroll area for the classes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)  # Set a reasonable minimum height

        # Create a widget to hold the grid layout
        scroll_content = QWidget()

        # Use grid layout instead of form layout for more efficient space usage
        self.class_grid = QGridLayout(scroll_content)
        self.class_grid.setColumnStretch(0, 0)  # ID column - minimal width
        self.class_grid.setColumnStretch(1, 1)  # Checkbox column - takes more space

        # Set proper spacing
        self.class_grid.setVerticalSpacing(8)
        self.class_grid.setHorizontalSpacing(10)

        # Add the scroll widget to the scroll area
        scroll_area.setWidget(scroll_content)

        # Store grid layout for _update_class_checkboxes method
        self.classes_grid_layout = self.class_grid

        # Add the scroll area to the classes layout
        classes_layout.addWidget(scroll_area)

        # Create checkboxes for all classes
        self.class_checkboxes = {}

        # Buttons for class selection
        buttons_layout = QHBoxLayout()

        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all_classes)
        select_all_button.setMinimumHeight(30)
        buttons_layout.addWidget(select_all_button)

        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(self.deselect_all_classes)
        deselect_all_button.setMinimumHeight(30)
        buttons_layout.addWidget(deselect_all_button)

        classes_layout.addLayout(buttons_layout)

        detection_layout.addWidget(self.classes_group)

        # Notification settings tab
        notification_tab = QWidget()
        notification_layout = QVBoxLayout(notification_tab)

        # Telegram settings
        telegram_group = QGroupBox("Telegram Notifications")
        telegram_layout = QFormLayout(telegram_group)

        self.telegram_bot_token = QLineEdit()
        telegram_layout.addRow("Bot Token:", self.telegram_bot_token)

        self.telegram_chat_id = QLineEdit()
        telegram_layout.addRow("Chat ID:", self.telegram_chat_id)

        self.telegram_thread_id = QSpinBox()
        self.telegram_thread_id.setRange(0, 999999999)
        self.telegram_thread_id.setSpecialValueText("None (Disabled)")
        telegram_layout.addRow("Thread ID:", self.telegram_thread_id)

        # Test button
        self.test_telegram_button = QPushButton("Test Telegram")
        self.test_telegram_button.clicked.connect(self.test_telegram)
        telegram_layout.addRow("", self.test_telegram_button)

        notification_layout.addWidget(telegram_group)

        # Email settings
        self.email_group = QGroupBox("Email Notifications")
        self.email_group.setCheckable(True)
        self.email_group.setChecked(False)
        email_layout = QFormLayout(self.email_group)

        self.email_smtp_server = QLineEdit()
        email_layout.addRow("SMTP Server:", self.email_smtp_server)

        self.email_smtp_port = QSpinBox()
        self.email_smtp_port.setRange(1, 65535)
        self.email_smtp_port.setValue(587)
        email_layout.addRow("SMTP Port:", self.email_smtp_port)

        self.email_use_ssl = QCheckBox("Use SSL/TLS")
        self.email_use_ssl.setChecked(True)
        email_layout.addRow("", self.email_use_ssl)

        self.email_username = QLineEdit()
        email_layout.addRow("Username:", self.email_username)

        self.email_password = QLineEdit()
        self.email_password.setEchoMode(QLineEdit.Password)
        email_layout.addRow("Password:", self.email_password)

        self.email_from = QLineEdit()
        email_layout.addRow("From Email:", self.email_from)

        self.email_to = QLineEdit()
        email_layout.addRow("To Email:", self.email_to)

        # Test button
        self.test_email_button = QPushButton("Test Email")
        self.test_email_button.clicked.connect(self.test_email)
        email_layout.addRow("", self.test_email_button)

        notification_layout.addWidget(self.email_group)

        # Add tabs
        self.tabs.addTab(general_tab, "General")
        self.tabs.addTab(detection_tab, "Detection")
        self.tabs.addTab(notification_tab, "Notifications")

        # Bottom buttons
        buttons_layout = QHBoxLayout()

        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        buttons_layout.addWidget(self.save_button)

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        buttons_layout.addWidget(self.reset_button)

        main_layout.addLayout(buttons_layout)

        # Update checkboxes - will be populated in update_settings
        self._update_class_checkboxes()

    def update_settings(self, settings: dict):
        """Update UI with settings values"""
        # Camera settings
        self.rtsp_url.setText(settings.get("rtsp_url", ""))
        self.resize_width.setValue(settings.get("resize_width", 640))
        self.resize_height.setValue(settings.get("resize_height", 480))
        self.auto_connect.setChecked(settings.get("auto_connect_camera", False))

        # RTSP transport settings
        transport = settings.get("rtsp_transport", "tcp")
        index = self.rtsp_transport.findData(transport)
        if index >= 0:
            self.rtsp_transport.setCurrentIndex(index)

        self.rtsp_auto_switch.setChecked(settings.get("rtsp_auto_switch", True))
        self.buffer_size.setValue(settings.get("buffer_size", 30))
        self.enable_adaptive_buffer.setChecked(settings.get("enable_adaptive_buffer", True))

        # Alert settings
        self.object_threshold.setValue(settings.get("object_threshold", 1))
        self.alert_cooldown.setValue(settings.get("alert_cooldown", 60))

        # Sound alert settings
        self.enable_sound_alert.setChecked(settings.get("enable_sound_alert", False))
        self.sound_alert_file.setText(settings.get("sound_alert_file", ""))

        # Model settings
        self.yolo_model_path.setText(settings.get("yolo_model_path", ""))
        self.use_gpu.setChecked(settings.get("use_gpu", True))
        self.yolo_confidence.setValue(settings.get("yolo_confidence_threshold", 0.25))

        # Classes of interest
        classes_of_interest = settings.get("classes_of_interest", list(range(40)))

        for class_id, checkbox in self.class_checkboxes.items():
            checkbox.setChecked(class_id in classes_of_interest)

        # Telegram settings
        self.telegram_bot_token.setText(settings.get("telegram_bot_token", ""))
        self.telegram_chat_id.setText(settings.get("telegram_chat_id", ""))
        self.telegram_thread_id.setValue(settings.get("telegram_message_thread_id", 0))

        # Email settings
        email_enabled = settings.get("email_enabled", False)
        self.email_group.setChecked(email_enabled)

        self.email_smtp_server.setText(settings.get("email_smtp_server", ""))
        self.email_smtp_port.setValue(settings.get("email_smtp_port", 587))
        self.email_use_ssl.setChecked(settings.get("email_use_ssl", True))
        self.email_username.setText(settings.get("email_username", ""))
        self.email_password.setText(settings.get("email_password", ""))
        self.email_from.setText(settings.get("email_from", ""))
        self.email_to.setText(settings.get("email_to", ""))

    def get_settings(self) -> dict:
        """Get settings from UI"""
        settings = {}

        # Classes of interest
        settings["classes_of_interest"] = [
            class_id for class_id, checkbox in self.class_checkboxes.items()
            if checkbox.isChecked()
        ]

        # Camera settings
        settings["rtsp_url"] = self.rtsp_url.text()
        settings["resize_width"] = self.resize_width.value()
        settings["resize_height"] = self.resize_height.value()
        settings["auto_connect_camera"] = self.auto_connect.isChecked()

        # RTSP transport settings
        settings["rtsp_transport"] = self.rtsp_transport.currentData()
        settings["rtsp_auto_switch"] = self.rtsp_auto_switch.isChecked()
        settings["buffer_size"] = self.buffer_size.value()
        settings["enable_adaptive_buffer"] = self.enable_adaptive_buffer.isChecked()

        # Alert settings
        settings["object_threshold"] = self.object_threshold.value()
        settings["alert_cooldown"] = self.alert_cooldown.value()

        # Sound alert settings
        settings["enable_sound_alert"] = self.enable_sound_alert.isChecked()
        settings["sound_alert_file"] = self.sound_alert_file.text()

        # Model settings
        settings["yolo_model_path"] = self.yolo_model_path.text()
        settings["use_gpu"] = self.use_gpu.isChecked()
        settings["yolo_confidence_threshold"] = self.yolo_confidence.value()

        # Classes of interest
        settings["classes_of_interest"] = [
            class_id for class_id, checkbox in self.class_checkboxes.items()
            if checkbox.isChecked()
        ]

        # Telegram settings
        settings["telegram_bot_token"] = self.telegram_bot_token.text()
        settings["telegram_chat_id"] = self.telegram_chat_id.text()
        settings["telegram_message_thread_id"] = self.telegram_thread_id.value()

        # Email settings
        settings["email_enabled"] = self.email_group.isChecked()

        settings["email_smtp_server"] = self.email_smtp_server.text()
        settings["email_smtp_port"] = self.email_smtp_port.value()
        settings["email_use_ssl"] = self.email_use_ssl.isChecked()
        settings["email_username"] = self.email_username.text()
        settings["email_password"] = self.email_password.text()
        settings["email_from"] = self.email_from.text()
        settings["email_to"] = self.email_to.text()

        return settings

    def _setup_priority_tab(self):
        """Setup the priority configuration tab"""
        # This is a placeholder for the priority tab setup
        # We'll implement this in a separate method later
        pass

    def save_settings(self):
        """Save settings to config"""
        settings = self.get_settings()

        # Update config
        for key, value in settings.items():
            self.config.set(key, value)

        # Save config to file
        if self.config.save():
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        else:
            QMessageBox.warning(self, "Settings", "Failed to save settings.")

    def reset_settings(self):
        """Reset settings to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.config.reset_to_defaults()
            self.update_settings(self.config.get_all())
            QMessageBox.information(self, "Settings", "Settings reset to defaults.")

    def browse_sound_file(self):
        """Browse for sound file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sound File",
            "",
            "Sound Files (*.mp3 *.wav);;All Files (*)"
        )

        if file_path:
            self.sound_alert_file.setText(file_path)

    def browse_model_file(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model File",
            "",
            "Model Files (*.pt *.pth *.weights);;All Files (*)"
        )

        if file_path:
            self.yolo_model_path.setText(file_path)

    def test_camera_connection(self):
        """Test camera connection"""
        import cv2

        rtsp_url = self.rtsp_url.text()
        transport = self.rtsp_transport.currentData()

        if not rtsp_url:
            QMessageBox.warning(self, "Test Camera", "Please enter an RTSP URL.")
            return

        # Show connecting message
        QMessageBox.information(self, "Test Camera",
                                f"Connecting to camera using {transport.upper()}...\nThis may take a few seconds.")

        try:
            # Try to open camera using OpenCV
            # For RTSP, we can set the transport protocol
            if rtsp_url.lower().startswith("rtsp://"):
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                # Set transport protocol (0=UDP, 1=TCP)
                transport_value = 1 if transport == "tcp" else 0
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, transport_value)
            else:
                cap = cv2.VideoCapture(rtsp_url)

            if not cap.isOpened():
                QMessageBox.critical(self, "Test Camera", f"Failed to connect to camera using {transport}.")
                return

            # Try to read a frame
            ret, frame = cap.read()

            if not ret:
                QMessageBox.critical(self, "Test Camera",
                                     f"Connected to camera but failed to read frame using {transport}.")
                cap.release()
                return

            # Success
            cap.release()
            QMessageBox.information(self, "Test Camera", f"Camera connection successful using {transport}!")

        except Exception as e:
            QMessageBox.critical(self, "Test Camera", f"Error connecting to camera: {e}")

    def test_sound(self):
        """Test sound alert"""
        sound_file = self.sound_alert_file.text()

        if not sound_file or not os.path.exists(sound_file):
            QMessageBox.warning(self, "Test Sound", "Please select a valid sound file.")
            return

        try:
            import pygame
            pygame.mixer.init()
            sound = pygame.mixer.Sound(sound_file)
            sound.play()
            QMessageBox.information(self, "Test Sound", "Playing sound...")
        except Exception as e:
            QMessageBox.critical(self, "Test Sound", f"Error playing sound: {e}")

    def test_telegram(self):
        """Test Telegram notification"""
        bot_token = self.telegram_bot_token.text()
        chat_id = self.telegram_chat_id.text()

        if not bot_token or not chat_id:
            QMessageBox.warning(self, "Test Telegram", "Please enter Bot Token and Chat ID.")
            return

        try:
            import requests

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            thread_id = self.telegram_thread_id.value() if self.telegram_thread_id.value() > 0 else None

            payload = {
                "chat_id": chat_id,
                "text": "FOD Detection System - Test Message"
            }

            if thread_id:
                payload["message_thread_id"] = thread_id

            response = requests.post(url, data=payload, timeout=10)

            if response.status_code == 200:
                QMessageBox.information(self, "Test Telegram", "Telegram test message sent successfully!")
            else:
                QMessageBox.critical(self, "Test Telegram", f"Failed to send Telegram message: {response.text}")

        except Exception as e:
            QMessageBox.critical(self, "Test Telegram", f"Error sending Telegram message: {e}")

    def test_email(self):
        """Test email notification"""
        smtp_server = self.email_smtp_server.text()
        smtp_port = self.email_smtp_port.value()
        username = self.email_username.text()
        password = self.email_password.text()
        from_addr = self.email_from.text()
        to_addr = self.email_to.text()
        use_ssl = self.email_use_ssl.isChecked()

        if not all([smtp_server, smtp_port, username, password, from_addr, to_addr]):
            QMessageBox.warning(self, "Test Email", "Please fill in all email settings.")
            return

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_addr
            msg['To'] = to_addr
            msg['Subject'] = "FOD Detection System - Test Email"

            body = "This is a test email from the FOD Detection System."
            msg.attach(MIMEText(body, 'plain'))

            # Connect to SMTP server
            if use_ssl:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()

            # Login
            server.login(username, password)

            # Send email
            server.sendmail(from_addr, to_addr, msg.as_string())

            # Close connection
            server.quit()

            QMessageBox.information(self, "Test Email", "Test email sent successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Test Email", f"Error sending email: {e}")

    def select_all_classes(self):
        """Select all classes"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(True)

    def deselect_all_classes(self):
        """Deselect all classes"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(False)

    def _get_class_names(self) -> dict:
        """Get dictionary of class names"""
        return {
            0: "AdjustableClamp", 1: "AdjustableWrench", 2: "Battery",
            3: "Bolt", 4: "BoltNutSet", 5: "BoltWasher",
            6: "ClampPart", 7: "Cutter", 8: "FuelCap",
            9: "Hammer", 10: "Hose", 11: "Label",
            12: "LuggagePart", 13: "LuggageTag", 14: "MetalPart",
            15: "MetalSheet", 16: "Nail", 17: "Nut",
            18: "PaintChip", 19: "Pen", 20: "PlasticPart",
            21: "Pliers", 22: "Rock", 23: "Screw",
            24: "Screwdriver", 25: "SodaCan", 26: "Tape",
            27: "Washer", 28: "Wire", 29: "Wood",
            30: "Wrench", 31: "Copper", 32: "Metallic shine",
            33: "Eyebolt", 34: "AsphaltCrack", 35: "FaucetHandle",
            36: "Tie-Wrap", 37: "Pitot cover", 38: "Scissors",
            39: "NutShell"
        }

    def _update_class_checkboxes(self):
        """Update the class checkboxes based on current class definitions"""
        # Clear existing checkboxes
        for widget in self.class_checkboxes.values():
            if widget.parent():
                layout = widget.parent().layout()
                if layout:
                    layout.removeWidget(widget)
            widget.deleteLater()

        self.class_checkboxes = {}

        # Get current classes
        if self.class_manager:
            class_names = {}
            for class_info in self.class_manager.get_all_classes():
                class_id = class_info["class_id"]
                class_names[class_id] = class_info["class_name"]
        else:
            class_names = self._get_class_names()

        # Get current classes of interest
        classes_of_interest = self.config.get("classes_of_interest", list(range(40)))

        # Check if we have the grid layout
        if not hasattr(self, 'classes_grid_layout'):
            logger.warning("Classes grid layout not found, skipping update")
            return

        # Clear the grid layout
        while self.classes_grid_layout.count():
            item = self.classes_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Sort class IDs for consistent display
        sorted_classes = sorted(class_names.items())

        # Determine optimal grid layout - calculate columns based on class count
        # Use more columns for more classes to make better use of space
        class_count = len(sorted_classes)
        cols = 1
        if class_count > 20:
            cols = 3
        elif class_count > 10:
            cols = 2

        # Add checkboxes to grid layout with labels
        for i, (class_id, class_name) in enumerate(sorted_classes):
            # Calculate row and column in grid
            row = i // cols
            col = i % cols * 2  # Each class takes 2 columns (label + checkbox)

            # Create ID label
            id_label = QLabel(f"{class_id}:")

            # Create checkbox with the class name
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(class_id in classes_of_interest)

            # Set font size explicitly to ensure readability
            font = checkbox.font()
            font.setPointSize(9)  # Adjust point size as needed
            checkbox.setFont(font)
            id_label.setFont(font)

            # Add to grid - labels in even columns, checkboxes in odd columns
            self.classes_grid_layout.addWidget(id_label, row, col)
            self.classes_grid_layout.addWidget(checkbox, row, col + 1)

            # Store checkbox reference
            self.class_checkboxes[class_id] = checkbox