import logging
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QPushButton, QLabel,
                             QComboBox, QCheckBox, QMessageBox, QDialogButtonBox,
                             QGroupBox, QFormLayout, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QFont

logger = logging.getLogger("FOD.ClassMappingDialog")


class ClassMappingDialog(QDialog):
    """
    Dialog for mapping classes between different models
    """

    def __init__(self, old_model, new_model, class_manager, parent=None):
        """
        Initialize the class mapping dialog

        Args:
            old_model: Old model name or path
            new_model: New model name or path
            class_manager: ClassManager instance
            parent: Parent widget
        """
        super().__init__(parent)

        self.old_model = self._extract_model_name(old_model)
        self.new_model = self._extract_model_name(new_model)
        self.class_manager = class_manager

        # Store mappings
        self.mappings = {}

        # Set window properties
        self.setWindowTitle("Class Mapping")
        self.setMinimumSize(800, 600)

        # Initialize UI
        self.init_ui()

        # Load classes and mappings
        self.load_classes()

    def _extract_model_name(self, model_path):
        """Extract model name from path"""
        import os
        return os.path.splitext(os.path.basename(model_path))[0]

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)

        # Header with instructions
        header = QLabel(
            f"Map classes between models:<br>"
            f"<b>{self.old_model}</b> → <b>{self.new_model}</b><br><br>"
            f"Classes with similar names will be mapped automatically. "
            f"Review and adjust the mappings as needed."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Options
        options_group = QGroupBox("Mapping Options")
        options_layout = QFormLayout(options_group)

        self.inherit_priority = QCheckBox("Inherit priorities from source classes")
        self.inherit_priority.setChecked(True)
        options_layout.addRow("", self.inherit_priority)

        self.similarity_threshold = QSpinBox()
        self.similarity_threshold.setRange(50, 100)
        self.similarity_threshold.setValue(70)
        self.similarity_threshold.setSuffix("%")
        options_layout.addRow("Similarity threshold:", self.similarity_threshold)

        layout.addWidget(options_group)

        # Mapping table
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(5)
        self.mapping_table.setHorizontalHeaderLabels([
            f"Source Class ID ({self.old_model})",
            "Source Class Name",
            f"Target Class ID ({self.new_model})",
            "Target Class Name",
            "Priority"
        ])
        self.mapping_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mapping_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.mapping_table)

        # Auto-map button
        auto_map_button = QPushButton("Auto-Map Classes")
        auto_map_button.clicked.connect(self.auto_map_classes)
        layout.addWidget(auto_map_button)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def load_classes(self):
        """Load classes from both models and existing mappings"""
        # Get all classes for both models
        self.source_classes = self.class_manager.get_classes_by_model(self.old_model)
        self.target_classes = self.class_manager.get_classes_by_model(self.new_model)

        # Load existing mappings
        mapping_key = f"{self.old_model}:{self.new_model}"
        self.mappings = self.class_manager.mapper.mappings.get(mapping_key, {})

        # Fill table with source classes
        self.mapping_table.setRowCount(len(self.source_classes))

        row = 0
        for source_id, source_info in self.source_classes.items():
            # Source class ID
            id_item = QTableWidgetItem(str(source_id))
            id_item.setData(Qt.UserRole, source_id)
            self.mapping_table.setItem(row, 0, id_item)

            # Source class name
            name_item = QTableWidgetItem(source_info["class_name"])
            if source_info.get("custom", False):
                name_item.setBackground(QBrush(QColor(240, 240, 255)))
            self.mapping_table.setItem(row, 1, name_item)

            # Target class combo box
            target_combo = QComboBox()
            target_combo.addItem("-- Not Mapped --", -1)

            # Add all target classes
            for target_id, target_info in self.target_classes.items():
                target_combo.addItem(
                    f"{target_id}: {target_info['class_name']}",
                    target_id
                )

            # Check if mapping exists
            if str(source_id) in self.mappings:
                target_id = int(self.mappings[str(source_id)])
                index = target_combo.findData(target_id)
                if index >= 0:
                    target_combo.setCurrentIndex(index)

            self.mapping_table.setCellWidget(row, 2, target_combo)

            # Target class name - will be updated when combo changes
            target_combo.currentIndexChanged.connect(
                lambda idx, r=row, c=target_combo: self.update_target_name(r, c)
            )
            self.update_target_name(row, target_combo)

            # Priority combo
            priority_combo = QComboBox()
            priority_combo.addItem("Low", 1)
            priority_combo.addItem("Medium", 2)
            priority_combo.addItem("High", 3)
            priority_combo.addItem("Critical", 4)

            # Set current priority from source class
            priority = source_info.get("priority", 1)
            index = priority_combo.findData(priority)
            if index >= 0:
                priority_combo.setCurrentIndex(index)

            self.mapping_table.setCellWidget(row, 4, priority_combo)

            row += 1

    def update_target_name(self, row, combo):
        """
        Update target name when combo selection changes

        Args:
            row: Table row
            combo: Combo box that changed
        """
        target_id = combo.currentData()

        if target_id < 0:
            # Not mapped
            name_item = QTableWidgetItem("Not mapped")
            name_item.setForeground(QBrush(QColor(150, 150, 150)))
        else:
            # Get name from target classes
            target_info = self.target_classes.get(target_id, {})
            name = target_info.get("class_name", f"Unknown-{target_id}")
            name_item = QTableWidgetItem(name)

            # Highlight custom classes
            if target_info.get("custom", False):
                name_item.setBackground(QBrush(QColor(240, 240, 255)))

        self.mapping_table.setItem(row, 3, name_item)

        # Update internal mappings
        source_item = self.mapping_table.item(row, 0)
        source_id = source_item.data(Qt.UserRole)

        if target_id >= 0:
            self.mappings[str(source_id)] = str(target_id)
        else:
            # Remove from mappings if exists
            if str(source_id) in self.mappings:
                del self.mappings[str(source_id)]

    def auto_map_classes(self):
        """Automatically map classes based on name similarity"""
        try:
            from difflib import SequenceMatcher

            # Get similarity threshold (as fraction)
            threshold = self.similarity_threshold.value() / 100.0

            # Track newly mapped classes
            new_mappings = 0

            # For each source class
            for row in range(self.mapping_table.rowCount()):
                # Skip if already mapped
                target_combo = self.mapping_table.cellWidget(row, 2)
                if target_combo.currentData() >= 0:
                    continue

                # Get source class details
                source_item = self.mapping_table.item(row, 0)
                source_id = source_item.data(Qt.UserRole)
                source_name = self.mapping_table.item(row, 1).text().lower()

                best_match = None
                best_ratio = 0.0

                # Find best matching target class
                for target_id, target_info in self.target_classes.items():
                    target_name = target_info["class_name"].lower()

                    # Skip if target already mapped (avoid duplicates)
                    target_already_mapped = False
                    for m_row in range(self.mapping_table.rowCount()):
                        if m_row == row:
                            continue
                        m_combo = self.mapping_table.cellWidget(m_row, 2)
                        if m_combo.currentData() == target_id:
                            target_already_mapped = True
                            break

                    if target_already_mapped:
                        continue

                    # Calculate similarity
                    ratio = SequenceMatcher(None, source_name, target_name).ratio()

                    # If exact match or best so far
                    if source_name == target_name:
                        best_match = target_id
                        break
                    elif ratio > best_ratio and ratio >= threshold:
                        best_ratio = ratio
                        best_match = target_id

                # If match found, set in combo box
                if best_match is not None:
                    index = target_combo.findData(best_match)
                    if index >= 0:
                        target_combo.setCurrentIndex(index)
                        new_mappings += 1

            if new_mappings > 0:
                QMessageBox.information(
                    self,
                    "Auto-Mapping Complete",
                    f"Automatically mapped {new_mappings} classes based on name similarity."
                )
            else:
                QMessageBox.information(
                    self,
                    "Auto-Mapping Complete",
                    "No new mappings were created. Try lowering the similarity threshold."
                )
        except Exception as e:
            logger.error(f"Error in auto-mapping: {e}")
            QMessageBox.warning(
                self,
                "Auto-Mapping Error",
                f"Error during auto-mapping: {str(e)}"
            )

    def accept(self):
        """Handle OK button - save mappings"""
        try:
            # Get all mappings from table
            mappings = {}
            priority_updates = []

            for row in range(self.mapping_table.rowCount()):
                # Get source class ID
                source_item = self.mapping_table.item(row, 0)
                source_id = source_item.data(Qt.UserRole)

                # Get target class ID from combo
                target_combo = self.mapping_table.cellWidget(row, 2)
                target_id = target_combo.currentData()

                # Skip if not mapped
                if target_id < 0:
                    continue

                # Add to mappings
                mappings[str(source_id)] = str(target_id)

                # Check if we should inherit priority
                if self.inherit_priority.isChecked():
                    # Get priority from combo
                    priority_combo = self.mapping_table.cellWidget(row, 4)
                    priority = priority_combo.currentData()

                    # Get current target priority
                    target_info = self.target_classes.get(target_id, {})
                    current_priority = target_info.get("priority", 1)

                    # If different, add to updates
                    if priority != current_priority:
                        priority_updates.append((target_id, priority))

            # Save mappings
            mapping_key = f"{self.old_model}:{self.new_model}"
            self.class_manager.mapper.mappings[mapping_key] = mappings
            self.class_manager.mapper.save_mappings()

            # Update priorities if needed
            for target_id, priority in priority_updates:
                target_info = self.target_classes.get(target_id, {})
                if target_info:
                    self.class_manager.add_or_update_class(
                        class_id=target_id,
                        class_name=target_info["class_name"],
                        priority=priority,
                        color=target_info.get("color", "#808080"),
                        description=target_info.get("description", ""),
                        model_name=target_info.get("model_name", self.new_model),
                        custom=target_info.get("custom", False)
                    )

            # Log results
            logger.info(f"Saved {len(mappings)} class mappings from {self.old_model} to {self.new_model}")
            if priority_updates:
                logger.info(f"Updated priorities for {len(priority_updates)} target classes")

            super().accept()
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save class mappings: {str(e)}"
            )

    @staticmethod
    def show_mapping_dialog(old_model, new_model, class_manager, parent=None):
        """
        Static method to show the mapping dialog

        Args:
            old_model: Old model name or path
            new_model: New model name or path
            class_manager: ClassManager instance
            parent: Parent widget

        Returns:
            True if mappings were created, False otherwise
        """
        dialog = ClassMappingDialog(old_model, new_model, class_manager, parent)
        result = dialog.exec_()

        return result == QDialog.Accepted