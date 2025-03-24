import os
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QPushButton, QLabel, QComboBox,
                             QLineEdit, QSpinBox, QMessageBox, QFileDialog,
                             QDialog, QFormLayout, QColorDialog, QHeaderView,
                             QTextEdit, QGroupBox, QCheckBox, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush

from storage.class_manager import ClassManager

logger = logging.getLogger("FOD.ClassEditor")


class ClassEditorDialog(QDialog):
    """Dialog for editing a single class definition"""

    def __init__(self, class_manager, class_id=None, parent=None):
        super().__init__(parent)
        self.class_manager = class_manager
        self.class_id = class_id
        self.is_new = class_id is None

        if self.is_new:
            self.setWindowTitle("Add New Class")
            self.class_details = {
                "class_id": self.class_manager.get_next_available_id(),
                "class_name": "",
                "priority": 1,
                "color": "#808080",
                "description": "",
                "model_name": "",
                "custom": True
            }
        else:
            self.setWindowTitle(f"Edit Class {class_id}")
            self.class_details = self.class_manager.get_class_details(class_id)
            if not self.class_details:
                self.class_details = {
                    "class_id": class_id,
                    "class_name": f"Class-{class_id}",
                    "priority": 1,
                    "color": "#808080",
                    "description": "",
                    "model_name": "",
                    "custom": True
                }

        self.init_ui()
        self.load_class_data()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QFormLayout(self)

        # Class ID
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 999)
        self.id_spin.setValue(self.class_details["class_id"])
        self.id_spin.setEnabled(self.is_new)  # Only enable for new classes
        layout.addRow("Class ID:", self.id_spin)

        # Class Name
        self.name_edit = QLineEdit()
        layout.addRow("Class Name:", self.name_edit)

        # Priority
        self.priority_combo = QComboBox()
        self.priority_combo.addItem("Low (1)", 1)
        self.priority_combo.addItem("Medium (2)", 2)
        self.priority_combo.addItem("High (3)", 3)
        self.priority_combo.addItem("Critical (4)", 4)
        layout.addRow("Priority:", self.priority_combo)

        # Color
        color_layout = QHBoxLayout()
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(24, 24)
        self.color_preview.setStyleSheet("background-color: #808080; border: 1px solid black;")
        color_layout.addWidget(self.color_preview)

        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)
        color_layout.addWidget(self.color_button)

        layout.addRow("Color:", color_layout)

        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        layout.addRow("Description:", self.description_edit)

        # Model
        self.model_edit = QLineEdit()
        layout.addRow("Model:", self.model_edit)

        # Custom flag
        self.custom_check = QCheckBox("Custom class (user-defined)")
        layout.addRow("", self.custom_check)

        # Buttons
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_class)
        button_layout.addWidget(self.save_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addRow("", button_layout)

        self.setMinimumWidth(400)

    def load_class_data(self):
        """Load class data into form fields"""
        self.name_edit.setText(self.class_details["class_name"])

        # Set priority
        priority_index = self.priority_combo.findData(self.class_details["priority"])
        if priority_index >= 0:
            self.priority_combo.setCurrentIndex(priority_index)

        # Set color
        color = self.class_details["color"]
        self.color_preview.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

        # Set description
        self.description_edit.setText(self.class_details["description"])

        # Set model
        self.model_edit.setText(self.class_details["model_name"])

        # Set custom flag
        self.custom_check.setChecked(self.class_details["custom"])

    def choose_color(self):
        """Open color chooser dialog"""
        current_color = QColor(self.class_details["color"])
        color = QColorDialog.getColor(current_color, self, "Select Class Color")

        if color.isValid():
            color_hex = color.name()
            self.class_details["color"] = color_hex
            self.color_preview.setStyleSheet(f"background-color: {color_hex}; border: 1px solid black;")

    def save_class(self):
        """Save class data to database"""
        # Get values from form
        class_id = self.id_spin.value()
        class_name = self.name_edit.text().strip()

        if not class_name:
            QMessageBox.warning(self, "Input Error", "Class name cannot be empty")
            return

        priority = self.priority_combo.currentData()
        color = self.class_details["color"]
        description = self.description_edit.toPlainText()
        model_name = self.model_edit.text()
        custom = self.custom_check.isChecked()

        # Save to database
        success = self.class_manager.add_or_update_class(
            class_id, class_name, priority, color, description, model_name, custom
        )

        if success:
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Failed to save class data")


class ClassEditorWidget(QWidget):
    """
    Widget for managing object class definitions
    """

    classes_changed = pyqtSignal()

    def __init__(self, class_manager=None, parent=None):
        super().__init__(parent)

        # Initialize class manager if not provided
        if class_manager is None:
            self.class_manager = ClassManager()
        else:
            self.class_manager = class_manager

        self.init_ui()
        self.load_classes()

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)

        # Create tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create editor tab
        editor_tab = QWidget()
        editor_layout = QVBoxLayout(editor_tab)

        # Table for viewing classes
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(5)
        self.class_table.setHorizontalHeaderLabels(["ID", "Name", "Priority", "Model", "Custom"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.class_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.class_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.class_table.setSelectionMode(QTableWidget.SingleSelection)
        self.class_table.doubleClicked.connect(self.edit_selected_class)
        editor_layout.addWidget(self.class_table)

        # Control buttons
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Add Class")
        self.add_button.clicked.connect(self.add_new_class)
        button_layout.addWidget(self.add_button)

        self.edit_button = QPushButton("Edit Class")
        self.edit_button.clicked.connect(self.edit_selected_class)
        button_layout.addWidget(self.edit_button)

        self.delete_button = QPushButton("Delete Class")
        self.delete_button.clicked.connect(self.delete_selected_class)
        button_layout.addWidget(self.delete_button)

        button_layout.addStretch()

        self.import_button = QPushButton("Import Classes")
        self.import_button.clicked.connect(self.import_classes)
        button_layout.addWidget(self.import_button)

        self.export_button = QPushButton("Export Classes")
        self.export_button.clicked.connect(self.export_classes)
        button_layout.addWidget(self.export_button)

        editor_layout.addLayout(button_layout)

        # Create model management tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)

        # Model detection group
        model_group = QGroupBox("Model Detection")
        model_group_layout = QFormLayout(model_group)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.model_path_edit)

        self.browse_model_button = QPushButton("Browse")
        self.browse_model_button.clicked.connect(self.browse_model)
        path_layout.addWidget(self.browse_model_button)

        model_group_layout.addRow("Model Path:", path_layout)

        self.model_scan_button = QPushButton("Scan Model for Classes")
        self.model_scan_button.clicked.connect(self.scan_model)
        model_group_layout.addRow("", self.model_scan_button)

        model_layout.addWidget(model_group)

        # Auto-detection options
        auto_group = QGroupBox("Auto-Detection Options")
        auto_layout = QVBoxLayout(auto_group)

        self.auto_detect_check = QCheckBox("Auto-detect classes when model changes")
        self.auto_detect_check.setChecked(True)
        auto_layout.addWidget(self.auto_detect_check)

        self.preserve_custom_check = QCheckBox("Preserve custom class definitions when new model is loaded")
        self.preserve_custom_check.setChecked(True)
        auto_layout.addWidget(self.preserve_custom_check)

        model_layout.addWidget(auto_group)
        model_layout.addStretch()

        # Add tabs
        self.tabs.addTab(editor_tab, "Class Editor")
        self.tabs.addTab(model_tab, "Model Management")

    def load_classes(self):
        """Load class definitions into the table"""
        classes = self.class_manager.get_all_classes()

        # Set row count
        self.class_table.setRowCount(len(classes))

        # Fill table
        for row, class_info in enumerate(classes):
            # ID
            id_item = QTableWidgetItem(str(class_info["class_id"]))
            self.class_table.setItem(row, 0, id_item)

            # Name
            name_item = QTableWidgetItem(class_info["class_name"])
            self.class_table.setItem(row, 1, name_item)

            # Priority
            priority_names = {
                1: "Low",
                2: "Medium",
                3: "High",
                4: "Critical"
            }
            priority_text = priority_names.get(class_info["priority"], str(class_info["priority"]))
            priority_item = QTableWidgetItem(priority_text)

            # Color code by priority
            if class_info["priority"] == 4:  # Critical
                priority_item.setBackground(QBrush(QColor(255, 200, 200)))
            elif class_info["priority"] == 3:  # High
                priority_item.setBackground(QBrush(QColor(255, 230, 200)))
            elif class_info["priority"] == 2:  # Medium
                priority_item.setBackground(QBrush(QColor(255, 255, 200)))

            self.class_table.setItem(row, 2, priority_item)

            # Model
            model_item = QTableWidgetItem(class_info["model_name"])
            self.class_table.setItem(row, 3, model_item)

            # Custom
            custom_item = QTableWidgetItem("Yes" if class_info["custom"] else "No")
            self.class_table.setItem(row, 4, custom_item)

            # Highlight custom classes
            if class_info["custom"]:
                color = QColor(230, 230, 250)  # Light purple background for custom classes
                for col in range(5):
                    self.class_table.item(row, col).setBackground(QBrush(color))

    def add_new_class(self):
        """Open dialog to add a new class"""
        dialog = ClassEditorDialog(self.class_manager, None, self)

        if dialog.exec_() == QDialog.Accepted:
            self.load_classes()
            self.classes_changed.emit()

    def edit_selected_class(self):
        """Edit the selected class"""
        selected_rows = self.class_table.selectedItems()

        if not selected_rows:
            QMessageBox.warning(self, "Selection Required", "Please select a class to edit")
            return

        # Get class ID from the first column of selected row
        row = selected_rows[0].row()
        class_id_item = self.class_table.item(row, 0)
        class_id = int(class_id_item.text())

        dialog = ClassEditorDialog(self.class_manager, class_id, self)

        if dialog.exec_() == QDialog.Accepted:
            self.load_classes()
            self.classes_changed.emit()

    def delete_selected_class(self):
        """Delete the selected class"""
        selected_rows = self.class_table.selectedItems()

        if not selected_rows:
            QMessageBox.warning(self, "Selection Required", "Please select a class to delete")
            return

        # Get class info from the selected row
        row = selected_rows[0].row()
        class_id_item = self.class_table.item(row, 0)
        class_id = int(class_id_item.text())
        class_name_item = self.class_table.item(row, 1)
        class_name = class_name_item.text()

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete class {class_id}: {class_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            success = self.class_manager.delete_class(class_id)

            if success:
                self.load_classes()
                self.classes_changed.emit()
            else:
                QMessageBox.critical(self, "Error", f"Failed to delete class {class_id}")

    def import_classes(self):
        """Import classes from a JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Classes",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        # Import classes
        added, updated, errors = self.class_manager.import_from_file(file_path)

        QMessageBox.information(
            self,
            "Import Results",
            f"Classes imported from {file_path}:\n"
            f"Added: {added}\n"
            f"Updated: {updated}\n"
            f"Errors: {errors}"
        )

        self.load_classes()
        self.classes_changed.emit()

    def export_classes(self):
        """Export classes to a JSON file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Classes",
            "class_definitions.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        # Ask if user wants to export custom classes only
        reply = QMessageBox.question(
            self,
            "Export Options",
            "Export custom classes only?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        custom_only = (reply == QMessageBox.Yes)

        # Export classes
        success = self.class_manager.export_to_file(file_path, custom_only)

        if success:
            QMessageBox.information(
                self,
                "Export Successful",
                f"Classes exported to {file_path}"
            )
        else:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export classes to {file_path}"
            )

    def browse_model(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "Model Files (*.pt *.pth *.weights);;All Files (*)"
        )

        if file_path:
            self.model_path_edit.setText(file_path)

    def scan_model(self):
        """Scan selected model for classes"""
        model_path = self.model_path_edit.text()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Required", "Please select a valid model file")
            return

        # Get model filename
        model_name = os.path.basename(model_path)

        try:
            # Attempt to load the model and get class names
            # This requires the YOLO library, so wrap in try/except
            try:
                from ultralytics import YOLO

                # Load model
                model = YOLO(model_path)

                # Get class names if available
                class_names = {}
                if hasattr(model, 'names'):
                    for idx, name in model.names.items():
                        class_names[int(idx)] = name

                # Get class count
                class_count = len(class_names)

                # Update class definitions
                preserve_custom = self.preserve_custom_check.isChecked()

                # Confirm with user
                msg = f"Found {class_count} classes in model {model_name}.\n\n"

                if class_count > 0:
                    # Show first 5 classes as example
                    msg += "Examples:\n"
                    for i, (idx, name) in enumerate(list(class_names.items())[:5]):
                        msg += f"- {idx}: {name}\n"

                    if len(class_names) > 5:
                        msg += f"- ... and {len(class_names) - 5} more\n\n"

                msg += "Do you want to update the class definitions database?"

                reply = QMessageBox.question(
                    self,
                    "Model Scan Results",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    result = self.class_manager.update_from_model(model_name, class_count, class_names)

                    if result:
                        QMessageBox.information(
                            self,
                            "Update Successful",
                            f"Updated class definitions from model {model_name}"
                        )
                        self.load_classes()
                        self.classes_changed.emit()
                    else:
                        QMessageBox.critical(
                            self,
                            "Update Failed",
                            f"Failed to update class definitions from model"
                        )

            except ImportError:
                QMessageBox.warning(
                    self,
                    "Library Missing",
                    "YOLO library not found. Cannot scan model directly.\n\n"
                    "You can still manually import class definitions from a JSON file."
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Scan Failed",
                f"Error scanning model: {str(e)}"
            )