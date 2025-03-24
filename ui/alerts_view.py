import os
import logging
import json
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView, QComboBox, QDateEdit, QGroupBox,
                             QFormLayout, QMessageBox, QMenu, QDialog,
                             QTabWidget, QLineEdit, QSpinBox, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QDate
from PyQt5.QtGui import QPixmap, QImage, QColor

from storage.database import AlertDatabase

logger = logging.getLogger("FOD.AlertsView")


class AlertDetailsDialog(QDialog):
    """
    Dialog to show alert details
    """

    def __init__(self, alert, parent=None):
        super().__init__(parent)
        self.alert = alert
        self.setWindowTitle(f"Alert Details - {alert['roi_name']}")
        self.setMinimumSize(800, 600)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)

        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Details tab
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)

        # Format timestamp
        try:
            timestamp = datetime.strptime(self.alert['timestamp'], "%Y-%m-%d %H:%M:%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = self.alert['timestamp']

        # Basic info
        info_group = QGroupBox("Alert Information")
        info_layout = QFormLayout(info_group)

        info_layout.addRow("Alert ID:", QLabel(str(self.alert['id'])))
        info_layout.addRow("ROI:", QLabel(self.alert['roi_name']))
        info_layout.addRow("Camera:", QLabel(self.alert['camera_id']))
        info_layout.addRow("Timestamp:", QLabel(formatted_time))

        # Severity
        severity_text = {
            1: "Low",
            2: "Medium",
            3: "High"
        }.get(self.alert['severity'], "Unknown")

        severity_label = QLabel(severity_text)
        severity_color = {
            1: "blue",
            2: "orange",
            3: "red"
        }.get(self.alert['severity'], "gray")
        severity_label.setStyleSheet(f"color: {severity_color}; font-weight: bold;")
        info_layout.addRow("Severity:", severity_label)

        # Add highest priority info if available
        if 'highest_priority' in self.alert:
            priority_label = QLabel(self.alert['highest_priority'])
            priority_label.setStyleSheet("font-weight: bold;")
            info_layout.addRow("Highest Priority:", priority_label)

        details_layout.addWidget(info_group)

        # Detections info
        detections_group = QGroupBox("Detections")
        detections_layout = QVBoxLayout(detections_group)

        # Parse class counts from JSON
        class_counts = {}
        try:
            if 'class_counts' in self.alert:
                class_counts = self.alert['class_counts']
            else:
                class_counts = json.loads(self.alert['alert_message'])
        except:
            pass

        # Create table for detections
        detections_table = QTableWidget()
        detections_table.setColumnCount(3)  # Added priority column
        detections_table.setHorizontalHeaderLabels(["Class", "Count", "Priority"])
        detections_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Fill table
        class_names = self._get_class_names()
        detections_table.setRowCount(len(class_counts) + 1)  # +1 for total

        # Get default priority map
        priority_map = {
            1: "Low",
            2: "Medium",
            3: "High",
            4: "Critical"
        }

        # Get default class priorities from Alert class
        from core.alert_manager import Alert
        class_priorities = Alert.DEFAULT_CLASS_PRIORITIES

        row = 0
        total_count = 0
        for class_id, count in class_counts.items():
            class_id = int(class_id)
            class_name = class_names.get(class_id, f"Unknown-{class_id}")

            detections_table.setItem(row, 0, QTableWidgetItem(class_name))
            detections_table.setItem(row, 1, QTableWidgetItem(str(count)))

            # Add priority
            priority = class_priorities.get(class_id, 1)
            priority_text = priority_map.get(priority, "Unknown")
            priority_item = QTableWidgetItem(priority_text)

            # Color code by priority
            if priority == 4:  # Critical
                priority_item.setBackground(QColor(255, 100, 100))
            elif priority == 3:  # High
                priority_item.setBackground(QColor(255, 180, 100))
            elif priority == 2:  # Medium
                priority_item.setBackground(QColor(255, 255, 100))

            detections_table.setItem(row, 2, priority_item)

            total_count += count
            row += 1

        # Add total row
        detections_table.setItem(row, 0, QTableWidgetItem("Total"))
        detections_table.setItem(row, 1, QTableWidgetItem(str(total_count)))
        detections_table.setItem(row, 2, QTableWidgetItem(""))

        # Make total row bold
        font = detections_table.item(row, 0).font()
        font.setBold(True)
        detections_table.item(row, 0).setFont(font)
        detections_table.item(row, 1).setFont(font)

        detections_layout.addWidget(detections_table)
        details_layout.addWidget(detections_group)

        # Image tab
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        image_layout.addWidget(self.image_label)

        # Load image if exists
        if self.alert['snapshot_path'] and os.path.exists(self.alert['snapshot_path']):
            pixmap = QPixmap(self.alert['snapshot_path'])
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        else:
            self.image_label.setText("No image available")

        # Add tabs
        tabs.addTab(details_tab, "Details")
        tabs.addTab(image_tab, "Image")

        # Bottom buttons
        button_layout = QHBoxLayout()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        # Add button layout
        layout.addLayout(button_layout)

    def _get_class_names(self):
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


class AlertsViewWidget(QWidget):
    """
    Widget to display and manage alert history
    """

    alert_selected = pyqtSignal(dict)

    def __init__(self, db: AlertDatabase, parent=None):
        super().__init__(parent)

        self.db = db
        self.current_page = 0
        self.page_size = 50
        self.total_alerts = 0
        self.current_alerts = []

        # Initialize UI
        self.init_ui()

        # Update alert count
        self.update_alert_count()

        # Load initial data
        self.refresh()

    def init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)

        # Filter controls
        filter_group = QGroupBox("Filter Alerts")
        filter_layout = QFormLayout(filter_group)

        # Date range
        date_layout = QHBoxLayout()

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-1))
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_date)

        filter_layout.addRow("Date Range:", date_layout)

        # Severity filter
        self.severity_combo = QComboBox()
        self.severity_combo.addItem("All Severities", None)
        self.severity_combo.addItem("High", 3)
        self.severity_combo.addItem("Medium", 2)
        self.severity_combo.addItem("Low", 1)
        filter_layout.addRow("Severity:", self.severity_combo)

        # Camera filter
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("All Cameras", None)
        self.camera_combo.addItem("Main Camera", "main")
        filter_layout.addRow("Camera:", self.camera_combo)

        # ROI index filter
        self.roi_spin = QSpinBox()
        self.roi_spin.setMinimum(-1)
        self.roi_spin.setMaximum(100)
        self.roi_spin.setValue(-1)
        self.roi_spin.setSpecialValueText("All ROIs")
        filter_layout.addRow("ROI Index:", self.roi_spin)

        # Filter button
        filter_button = QPushButton("Apply Filter")
        filter_button.clicked.connect(self.refresh)
        filter_layout.addRow("", filter_button)

        main_layout.addWidget(filter_group)

        # Alert table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(6)
        self.alerts_table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "ROI", "Camera", "Severity", "Objects"
        ])
        self.alerts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.alerts_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.alerts_table.setSelectionMode(QTableWidget.SingleSelection)
        self.alerts_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.alerts_table.doubleClicked.connect(self.show_alert_details)
        self.alerts_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.alerts_table.customContextMenuRequested.connect(self.show_context_menu)

        main_layout.addWidget(self.alerts_table)

        # Pagination controls
        pagination_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_page)
        pagination_layout.addWidget(self.prev_button)

        self.page_label = QLabel("Page 1")
        pagination_layout.addWidget(self.page_label)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_page)
        pagination_layout.addWidget(self.next_button)

        self.total_label = QLabel("Total: 0 alerts")
        pagination_layout.addWidget(self.total_label)

        pagination_layout.addStretch()

        # Control buttons
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        pagination_layout.addWidget(self.refresh_button)

        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        pagination_layout.addWidget(self.export_button)

        main_layout.addLayout(pagination_layout)

    def refresh(self):
        """Refresh alert data based on filters"""
        # Get filter values
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")

        severity = self.severity_combo.currentData()
        camera_id = self.camera_combo.currentData()

        roi_index = self.roi_spin.value()
        if roi_index < 0:
            roi_index = None

        # Calculate offset
        offset = self.current_page * self.page_size

        # Get data from database
        self.current_alerts = self.db.get_alerts(
            limit=self.page_size,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            roi_index=roi_index,
            camera_id=camera_id,
            severity=severity
        )

        # Update table
        self.update_table()

        # Update pagination controls
        self.update_pagination()

    def update_table(self):
        """Update the alerts table with current data"""
        self.alerts_table.setRowCount(len(self.current_alerts))

        for row, alert in enumerate(self.current_alerts):
            # ID
            self.alerts_table.setItem(row, 0, QTableWidgetItem(str(alert['id'])))

            # Timestamp
            try:
                timestamp = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M:%S")
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = alert['timestamp']

            self.alerts_table.setItem(row, 1, QTableWidgetItem(formatted_time))

            # ROI
            roi_text = f"{alert['roi_name']} ({alert['roi_index']})"
            self.alerts_table.setItem(row, 2, QTableWidgetItem(roi_text))

            # Camera
            self.alerts_table.setItem(row, 3, QTableWidgetItem(alert['camera_id']))

            # Severity
            severity_text = {
                1: "Low",
                2: "Medium",
                3: "High"
            }.get(alert['severity'], "Unknown")

            severity_item = QTableWidgetItem(severity_text)

            # Set background color based on severity
            color = {
                1: QColor(200, 230, 255),  # Light blue
                2: QColor(255, 235, 156),  # Light orange
                3: QColor(255, 180, 180)  # Light red
            }.get(alert['severity'], QColor(240, 240, 240))

            severity_item.setBackground(color)
            self.alerts_table.setItem(row, 4, severity_item)

            # Count objects
            total_objects = 0
            try:
                if 'class_counts' in alert:
                    total_objects = sum(alert['class_counts'].values())
                else:
                    class_counts = json.loads(alert['alert_message'])
                    total_objects = sum(int(count) for count in class_counts.values())
            except:
                pass

            self.alerts_table.setItem(row, 5, QTableWidgetItem(str(total_objects)))

    def update_pagination(self):
        """Update pagination controls"""
        self.page_label.setText(f"Page {self.current_page + 1}")

        # Update alert count if needed
        self.update_alert_count()

        # Update total alerts label
        self.total_label.setText(f"Total: {self.total_alerts} alerts")

        # Enable/disable pagination buttons
        self.prev_button.setEnabled(self.current_page > 0)

        # Only enable next if we got a full page of results
        self.next_button.setEnabled(len(self.current_alerts) >= self.page_size)

    def update_alert_count(self):
        """Update the total alert count"""
        self.total_alerts = self.db.get_alert_count()

    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh()

    def next_page(self):
        """Go to next page"""
        self.current_page += 1
        self.refresh()

    def show_alert_details(self, index):
        """Show details for the selected alert"""
        row = index.row()
        if 0 <= row < len(self.current_alerts):
            alert = self.current_alerts[row]
            dialog = AlertDetailsDialog(alert, self)
            dialog.exec_()

    def show_context_menu(self, position):
        """Show context menu for right-click on table"""
        index = self.alerts_table.indexAt(position)

        if not index.isValid():
            return

        row = index.row()
        if row < 0 or row >= len(self.current_alerts):
            return

        alert = self.current_alerts[row]

        menu = QMenu(self)

        view_action = menu.addAction("View Details")
        view_action.triggered.connect(lambda: self.show_alert_details(index))

        menu.addSeparator()

        # Open snapshot if exists
        if alert['snapshot_path'] and os.path.exists(alert['snapshot_path']):
            open_image_action = menu.addAction("Open Image")
            open_image_action.triggered.connect(lambda: self.open_file(alert['snapshot_path']))

        # Open video if exists
        if alert['video_path'] and os.path.exists(alert['video_path']):
            open_video_action = menu.addAction("Open Video")
            open_video_action.triggered.connect(lambda: self.open_file(alert['video_path']))

        menu.addSeparator()

        delete_action = menu.addAction("Delete Alert")
        delete_action.triggered.connect(lambda: self.delete_alert(alert['id']))

        menu.exec_(self.alerts_table.viewport().mapToGlobal(position))

    def open_file(self, file_path):
        """Open a file with the default application"""
        import subprocess
        import platform
        import os

        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', file_path))
            else:  # Linux
                subprocess.call(('xdg-open', file_path))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")

    def delete_alert(self, alert_id):
        """Delete an alert from the database"""
        reply = QMessageBox.question(
            self,
            "Delete Alert",
            f"Are you sure you want to delete alert {alert_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            success = self.db.delete_alert(alert_id)

            if success:
                QMessageBox.information(self, "Success", f"Alert {alert_id} deleted.")
                self.refresh()
            else:
                QMessageBox.warning(self, "Error", f"Failed to delete alert {alert_id}.")

    def export_csv(self):
        """Export alerts to CSV file"""
        from PyQt5.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Alerts to CSV",
            "alerts_export.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            success = self.db.export_to_csv(file_path)

            if success:
                QMessageBox.information(self, "Export Successful", f"Alerts exported to {file_path}")
            else:
                QMessageBox.warning(self, "Export Failed", "Failed to export alerts to CSV.")