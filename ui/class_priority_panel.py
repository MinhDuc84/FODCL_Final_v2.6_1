import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QGroupBox, QFormLayout, QLabel, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox, QPushButton,
                             QMessageBox, QColorDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QFont

logger = logging.getLogger("FOD.ClassPriorityPanel")


class ClassPriorityPanel(QWidget):
    """
    Widget for managing class priorities
    """

    priorities_changed = pyqtSignal()

    def __init__(self, class_manager, config_manager, parent=None):
        """
        Initialize the class priority panel

        Args:
            class_manager: ClassManager instance
            config_manager: ConfigManager instance
            parent: Parent widget
        """
        super().__init__(parent)

        self.class_manager = class_manager
        self.config_manager = config_manager

        # Priority levels
        self.priority_levels = [
            {"name": "Low", "value": 1, "color": QColor(220, 230, 255)},  # Light blue
            {"name": "Medium", "value": 2, "color": QColor(255, 240, 200)},  # Light yellow
            {"name": "High", "value": 3, "color": QColor(255, 200, 180)},  # Light orange
            {"name": "Critical", "value": 4, "color": QColor(255, 160, 160)}  # Light red
        ]

        # Initialize UI
        self.init_ui()

        # Register for class changes
        if hasattr(self.class_manager, "add_listener"):
            self.class_manager.add_listener(self._handle_class_change)

        # Load classes
        self.load_classes()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Set priority levels for each object class. Higher priority objects will "
            "trigger higher severity alerts and notifications."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Create table for classes
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(4)
        self.class_table.setHorizontalHeaderLabels(["ID", "Class Name", "Priority", "Color"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.class_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.class_table.setSortingEnabled(True)
        layout.addWidget(self.class_table)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.set_all_button = QPushButton("Set All Selected...")
        self.set_all_button.clicked.connect(self.set_priority_for_selected)
        controls_layout.addWidget(self.set_all_button)

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        controls_layout.addWidget(self.reset_button)

        controls_layout.addStretch()

        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_changes)
        controls_layout.addWidget(self.apply_button)

        layout.addLayout(controls_layout)

        # Filter controls
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel("Filter by:"))

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Classes", None)
        self.filter_combo.addItem("Low Priority", 1)
        self.filter_combo.addItem("Medium Priority", 2)
        self.filter_combo.addItem("High Priority", 3)
        self.filter_combo.addItem("Critical Priority", 4)
        self.filter_combo.addItem("Custom Classes", "custom")
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_combo)

        filter_layout.addStretch()

        # Model info
        self.model_info = QLabel("Current Model: Not loaded")
        filter_layout.addWidget(self.model_info)

        layout.addLayout(filter_layout)

    def load_classes(self):
        """Load classes into the table"""
        # Clear table
        self.class_table.setRowCount(0)

        # Get all classes
        classes = self.class_manager.get_all_classes()

        if not classes:
            return

        # Populate table
        self.class_table.setRowCount(len(classes))

        for row, class_info in enumerate(classes):
            class_id = class_info["class_id"]
            class_name = class_info["class_name"]
            priority = class_info["priority"]
            color = class_info["color"]
            custom = class_info["custom"]

            # ID column
            id_item = QTableWidgetItem(str(class_id))
            id_item.setData(Qt.UserRole, class_id)
            self.class_table.setItem(row, 0, id_item)

            # Name column
            name_item = QTableWidgetItem(class_name)
            if custom:
                font = name_item.font()
                font.setBold(True)
                name_item.setFont(font)
            self.class_table.setItem(row, 1, name_item)

            # Priority column - use combo box
            priority_combo = QComboBox()
            for level in self.priority_levels:
                priority_combo.addItem(level["name"], level["value"])

            # Set current priority
            index = priority_combo.findData(priority)
            if index >= 0:
                priority_combo.setCurrentIndex(index)

            # Set background color based on priority
            if priority > 0 and priority <= len(self.priority_levels):
                priority_combo.setStyleSheet(
                    f"background-color: {self.priority_levels[priority - 1]['color'].name()}"
                )

            self.class_table.setCellWidget(row, 2, priority_combo)

            # Color column - show color and button to change
            color_button = QPushButton()
            if color.startswith("#"):
                qcolor = QColor(color)
            else:
                qcolor = QColor(255, 0, 0)  # Default to red

            color_button.setStyleSheet(f"background-color: {qcolor.name()}")
            color_button.clicked.connect(lambda checked, r=row: self.choose_color(r))

            self.class_table.setCellWidget(row, 3, color_button)

            # Mark custom classes with a different background
            if custom:
                for col in range(4):
                    item = self.class_table.item(row, col)
                    if item:
                        item.setBackground(QBrush(QColor(240, 240, 255)))

        # Update model info
        current_model = self.config_manager.get("yolo_model_path", "Not loaded")
        self.model_info.setText(f"Current Model: {current_model}")

    def choose_color(self, row):
        """
        Show color dialog for changing class color

        Args:
            row: Table row
        """
        try:
            # Get current color
            color_button = self.class_table.cellWidget(row, 3)
            current_color = QColor(color_button.styleSheet().split(":")[1].strip())

            # Open color dialog
            color = QColorDialog.getColor(current_color, self, "Select Class Color")

            # Update if valid
            if color.isValid():
                color_button.setStyleSheet(f"background-color: {color.name()}")
        except Exception as e:
            logger.error(f"Error choosing color: {e}")

    def apply_filter(self):
        """Apply filter to class table"""
        filter_value = self.filter_combo.currentData()

        for row in range(self.class_table.rowCount()):
            # Get priority for this row
            priority_combo = self.class_table.cellWidget(row, 2)
            priority = priority_combo.currentData()

            # Get custom status
            id_item = self.class_table.item(row, 0)
            class_id = id_item.data(Qt.UserRole)
            class_info = self.class_manager.get_class_details(class_id)
            is_custom = class_info.get("custom", False) if class_info else False

            # Determine visibility
            if filter_value is None:
                # Show all
                self.class_table.setRowHidden(row, False)
            elif filter_value == "custom":
                # Show only custom classes
                self.class_table.setRowHidden(row, not is_custom)
            else:
                # Show only matching priority
                self.class_table.setRowHidden(row, priority != filter_value)

    def set_priority_for_selected(self):
        """Set priority for all selected rows"""
        selected_rows = self.class_table.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select at least one class")
            return

        # Create list of unique rows
        rows = set(index.row() for index in selected_rows)

        # Create a dialog for priority selection
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Set Priority")
        layout = QVBoxLayout(dialog)

        # Create combo box in the dialog
        priority_combo = QComboBox()
        for level in self.priority_levels:
            priority_combo.addItem(level["name"], level["value"])
        layout.addWidget(priority_combo)

        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Show the dialog
        if dialog.exec_() == QDialog.Accepted:
            new_priority = priority_combo.currentData()
            priority_name = priority_combo.currentText()

            # Update all selected rows
            for row in rows:
                combo = self.class_table.cellWidget(row, 2)
                index = combo.findData(new_priority)
                if index >= 0:
                    combo.setCurrentIndex(index)

                    # Update background color
                    level_index = new_priority - 1
                    if 0 <= level_index < len(self.priority_levels):
                        combo.setStyleSheet(
                            f"background-color: {self.priority_levels[level_index]['color'].name()}"
                        )

            QMessageBox.information(
                self,
                "Priority Updated",
                f"Set priority to {priority_name} for {len(rows)} classes"
            )

    def reset_to_defaults(self):
        """Reset priorities to default values"""
        reply = QMessageBox.question(
            self,
            "Reset Priorities",
            "Reset all class priorities to default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Get default priorities from Alert class
        from core.alert_manager import Alert
        default_priorities = Alert.DEFAULT_CLASS_PRIORITIES

        # Update priorities in table
        for row in range(self.class_table.rowCount()):
            # Get class ID
            id_item = self.class_table.item(row, 0)
            class_id = id_item.data(Qt.UserRole)

            # Get default priority for this class
            default_priority = default_priorities.get(class_id, 1)  # Default to Low

            # Update combo box
            priority_combo = self.class_table.cellWidget(row, 2)
            index = priority_combo.findData(default_priority)
            if index >= 0:
                priority_combo.setCurrentIndex(index)

                # Update background color
                level_index = default_priority - 1
                if 0 <= level_index < len(self.priority_levels):
                    priority_combo.setStyleSheet(
                        f"background-color: {self.priority_levels[level_index]['color'].name()}"
                    )

        QMessageBox.information(self, "Reset Complete", "All class priorities reset to default values")

    def apply_changes(self):
        """Apply changes to class priorities"""
        try:
            changes = 0

            for row in range(self.class_table.rowCount()):
                # Get class ID
                id_item = self.class_table.item(row, 0)
                class_id = id_item.data(Qt.UserRole)

                # Get current class info
                class_info = self.class_manager.get_class_details(class_id)
                if not class_info:
                    continue

                # Get new priority value
                priority_combo = self.class_table.cellWidget(row, 2)
                new_priority = priority_combo.currentData()

                # Get new color
                color_button = self.class_table.cellWidget(row, 3)
                style = color_button.styleSheet()
                new_color = style.split(":")[1].strip()

                # Check if changed
                if (new_priority != class_info["priority"] or
                        new_color != class_info["color"]):
                    # Update class
                    self.class_manager.add_or_update_class(
                        class_id=class_id,
                        class_name=class_info["class_name"],
                        priority=new_priority,
                        color=new_color,
                        description=class_info.get("description", ""),
                        model_name=class_info.get("model_name", ""),
                        custom=class_info.get("custom", False)
                    )

                    changes += 1

            # Update configuration for syncing between components
            if changes > 0:
                # Emit signal that priorities changed
                self.priorities_changed.emit()

                QMessageBox.information(
                    self,
                    "Changes Applied",
                    f"Updated priorities for {changes} classes"
                )
        except Exception as e:
            logger.error(f"Error applying priority changes: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to update class priorities: {str(e)}"
            )

    def _handle_class_change(self, event):
        """
        Handle class change events

        Args:
            event: ClassChangeEvent from ClassManager
        """
        # Reload the class table if classes changed
        if event.action in ["add", "update", "delete", "import", "model_update"]:
            self.load_classes()