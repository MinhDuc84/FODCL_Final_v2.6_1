import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QListWidget, QListWidgetItem,
                             QDialog, QFormLayout, QLineEdit, QDoubleSpinBox,
                             QColorDialog, QSpinBox, QMessageBox, QSplitter,
                             QGroupBox, QCheckBox, QComboBox, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from core.video_source import VideoSource
from core.roi_manager import ROIManager, ROI
from core.detector import YOLODetector  # Import for class names only

logger = logging.getLogger("FOD.ROIEditor")


class ROIEditorWidget(QWidget):
    """
    Widget for creating and editing Regions of Interest (ROIs)
    """

    # Signal emitted when ROIs are changed
    rois_changed = pyqtSignal()

    def __init__(self, video_source: VideoSource, roi_manager: ROIManager, parent=None):
        super().__init__(parent)

        self.video_source = video_source
        self.roi_manager = roi_manager

        # ROI editing state
        self.creating_roi = False
        self.editing_roi = False
        self.selected_roi_index = None
        self.selected_point_index = None
        self.dragging = False
        self.resizing = False

        # Add missing variables for resizing
        self.resizing_start_point = None
        self.resizing_original_points = None
        self.resizing_center = None

        # Shape-specific drawing states
        self.rectangle_dragging = False
        self.circle_dragging = False
        self.freehand_drawing = False
        self.temp_second_point = None
        self.temp_radius_point = None

        # Display frame and temporary frame for editing
        self.current_frame = None
        self.editor_frame = None

        # Setup UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        # Create main layout
        main_layout = QHBoxLayout(self)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: ROI list and properties
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # ROI list
        roi_group = QGroupBox("ROI List")
        roi_list_layout = QVBoxLayout(roi_group)

        self.roi_list = QListWidget()
        self.roi_list.setMinimumWidth(200)
        self.roi_list.currentRowChanged.connect(self.on_roi_selected)
        roi_list_layout.addWidget(self.roi_list)

        # ROI list buttons
        list_buttons_layout = QHBoxLayout()

        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.clicked.connect(self.start_roi_creation)
        list_buttons_layout.addWidget(self.add_roi_button)

        self.edit_roi_button = QPushButton("Edit")
        self.edit_roi_button.clicked.connect(self.edit_selected_roi)
        list_buttons_layout.addWidget(self.edit_roi_button)

        self.delete_roi_button = QPushButton("Delete")
        self.delete_roi_button.clicked.connect(self.delete_selected_roi)
        list_buttons_layout.addWidget(self.delete_roi_button)

        roi_list_layout.addLayout(list_buttons_layout)
        left_layout.addWidget(roi_group)

        # Add shape selection toolbar
        shape_group = QGroupBox("ROI Shape")
        shape_layout = QHBoxLayout(shape_group)

        self.shape_polygon = QPushButton("Polygon")
        self.shape_polygon.setCheckable(True)
        self.shape_polygon.setChecked(True)
        shape_layout.addWidget(self.shape_polygon)

        self.shape_rectangle = QPushButton("Rectangle")
        self.shape_rectangle.setCheckable(True)
        shape_layout.addWidget(self.shape_rectangle)

        self.shape_circle = QPushButton("Circle")
        self.shape_circle.setCheckable(True)
        shape_layout.addWidget(self.shape_circle)

        self.shape_freehand = QPushButton("Freehand")
        self.shape_freehand.setCheckable(True)
        shape_layout.addWidget(self.shape_freehand)

        self.shape_bezier = QPushButton("Bezier Curve")
        self.shape_bezier.setCheckable(True)
        shape_layout.addWidget(self.shape_bezier)

        # Connect shape buttons to handler
        for btn in [self.shape_polygon, self.shape_rectangle,
                    self.shape_circle, self.shape_freehand, self.shape_bezier]:
            btn.clicked.connect(self.on_shape_selected)

        left_layout.addWidget(shape_group)

        # ROI properties
        properties_group = QGroupBox("ROI Properties")
        properties_layout = QFormLayout(properties_group)

        self.roi_name_edit = QLineEdit()
        properties_layout.addRow("Name:", self.roi_name_edit)

        self.roi_threshold_spin = QDoubleSpinBox()
        self.roi_threshold_spin.setMinimum(0.1)
        self.roi_threshold_spin.setMaximum(100)
        self.roi_threshold_spin.setSingleStep(0.1)
        self.roi_threshold_spin.setValue(1.0)
        properties_layout.addRow("Threshold:", self.roi_threshold_spin)

        self.roi_cooldown_spin = QSpinBox()
        self.roi_cooldown_spin.setMinimum(1)
        self.roi_cooldown_spin.setMaximum(3600)
        self.roi_cooldown_spin.setSingleStep(5)
        self.roi_cooldown_spin.setValue(60)
        properties_layout.addRow("Cooldown (s):", self.roi_cooldown_spin)

        self.roi_color_button = QPushButton()
        self.roi_color_button.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.roi_color_button.clicked.connect(self.select_roi_color)
        properties_layout.addRow("Color:", self.roi_color_button)

        # Classes filter
        classes_layout = QVBoxLayout()
        self.use_global_classes = QCheckBox("Use global classes")
        self.use_global_classes.setChecked(True)
        self.use_global_classes.stateChanged.connect(self.toggle_classes_filter)
        classes_layout.addWidget(self.use_global_classes)

        self.classes_combo = QComboBox()
        self.classes_combo.setEnabled(False)
        self.classes_combo.addItem("All Classes")

        # Get class names dynamically
        self.update_class_combo()
        classes_layout.addWidget(self.classes_combo)

        self.add_class_button = QPushButton("Add Class")
        self.add_class_button.setEnabled(False)
        self.add_class_button.clicked.connect(self.add_roi_class)
        classes_layout.addWidget(self.add_class_button)

        self.selected_classes_list = QListWidget()
        self.selected_classes_list.setEnabled(False)
        self.selected_classes_list.setMaximumHeight(100)
        classes_layout.addWidget(self.selected_classes_list)

        properties_layout.addRow("Classes:", classes_layout)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_roi_properties)
        properties_layout.addRow("", self.apply_button)

        left_layout.addWidget(properties_group)

        # Right panel: Camera view
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Instructions
        instructions = QLabel("Left-click to add points. Right-click to complete ROI.")
        instructions.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(instructions)

        # Camera view
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(960, 540)
        self.camera_view.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.camera_view)

        # Add zoom controls in the right panel
        zoom_controls = QHBoxLayout()

        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedSize(30, 30)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_controls.addWidget(self.zoom_out_button)

        self.zoom_level = QLabel("100%")
        zoom_controls.addWidget(self.zoom_level)

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setFixedSize(30, 30)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        zoom_controls.addWidget(self.zoom_in_button)

        self.zoom_reset_button = QPushButton("Reset")
        self.zoom_reset_button.clicked.connect(self.zoom_reset)
        zoom_controls.addWidget(self.zoom_reset_button)

        # Pan controls
        self.pan_mode_button = QPushButton("Pan Mode")
        self.pan_mode_button.setCheckable(True)
        zoom_controls.addWidget(self.pan_mode_button)

        right_layout.addLayout(zoom_controls)

        # Add sensitivity visualization controls
        sensitivity_layout = QHBoxLayout()

        self.sensitivity_check = QCheckBox("Show Sensitivity Preview")
        self.sensitivity_check.setChecked(False)
        self.sensitivity_check.stateChanged.connect(self.toggle_sensitivity_preview)
        sensitivity_layout.addWidget(self.sensitivity_check)

        self.threshold_min = QDoubleSpinBox()
        self.threshold_min.setRange(0.1, 0.9)
        self.threshold_min.setSingleStep(0.05)
        self.threshold_min.setValue(0.2)
        self.threshold_min.valueChanged.connect(self.update_sensitivity_preview)
        sensitivity_layout.addWidget(QLabel("Min:"))
        sensitivity_layout.addWidget(self.threshold_min)

        self.threshold_max = QDoubleSpinBox()
        self.threshold_max.setRange(0.1, 1.0)
        self.threshold_max.setSingleStep(0.05)
        self.threshold_max.setValue(0.8)
        self.threshold_max.valueChanged.connect(self.update_sensitivity_preview)
        sensitivity_layout.addWidget(QLabel("Max:"))
        sensitivity_layout.addWidget(self.threshold_max)

        right_layout.addLayout(sensitivity_layout)

        # Add alignment tools
        alignment_group = QGroupBox("Alignment Tools")
        alignment_layout = QHBoxLayout(alignment_group)

        self.enable_grid = QCheckBox("Show Grid")
        self.enable_grid.setChecked(False)
        self.enable_grid.stateChanged.connect(self.toggle_grid)
        alignment_layout.addWidget(self.enable_grid)

        self.grid_size = QSpinBox()
        self.grid_size.setRange(10, 100)
        self.grid_size.setValue(50)
        self.grid_size.setSingleStep(10)
        self.grid_size.setPrefix("Grid Size: ")
        self.grid_size.setSuffix("px")
        self.grid_size.valueChanged.connect(self.update_grid)
        alignment_layout.addWidget(self.grid_size)

        self.enable_snap = QCheckBox("Snap to Grid")
        self.enable_snap.setChecked(False)
        alignment_layout.addWidget(self.enable_snap)

        self.enable_rulers = QCheckBox("Show Rulers")
        self.enable_rulers.setChecked(False)
        self.enable_rulers.stateChanged.connect(lambda: self.update_frame(self.current_frame))
        alignment_layout.addWidget(self.enable_rulers)

        right_layout.addWidget(alignment_group)

        # Editing controls
        edit_controls = QHBoxLayout()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_roi_edit)
        edit_controls.addWidget(self.cancel_button)

        self.complete_button = QPushButton("Complete ROI")
        self.complete_button.clicked.connect(self.complete_roi)
        edit_controls.addWidget(self.complete_button)

        edit_controls.addStretch()

        self.mode_label = QLabel("Mode: Viewing")
        edit_controls.addWidget(self.mode_label)

        right_layout.addLayout(edit_controls)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([300, 700])

        # Initialize alignment properties
        self.show_grid = False
        self.grid_visible = False

        # Initialize zoom state
        self.zoom_factor = 1.0
        self.zoom_center = None
        self.panning = False
        self.pan_start = None
        self.pan_offset = (0, 0)
        self.sensitivity_preview_mode = False

        # Make the camera view accept mouse events
        self.camera_view.setMouseTracking(True)
        self.camera_view.mousePressEvent = self.on_mouse_press
        self.camera_view.mouseMoveEvent = self.on_mouse_move
        self.camera_view.mouseReleaseEvent = self.on_mouse_release

        # Add transparency control
        transparency_layout = QHBoxLayout()
        transparency_layout.addWidget(QLabel("ROI Transparency:"))

        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(1, 10)  # 1 = 10% opacity, 10 = 100% opacity
        self.transparency_slider.setValue(3)  # Default to 30%
        self.transparency_slider.setTickPosition(QSlider.TicksBelow)
        self.transparency_slider.setTickInterval(1)
        self.transparency_slider.valueChanged.connect(self.update_frame)
        transparency_layout.addWidget(self.transparency_slider)

        transparency_layout.addWidget(QLabel("Transparent"))
        transparency_layout.addStretch()
        transparency_layout.addWidget(QLabel("Opaque"))

        right_layout.addLayout(transparency_layout)

        # Update the ROI list
        self.refresh_roi_list()

    def on_shape_selected(self):
        """Handle ROI shape selection"""
        # Uncheck all other shape buttons
        sender = self.sender()

        for btn in [self.shape_polygon, self.shape_rectangle,
                    self.shape_circle, self.shape_freehand, self.shape_bezier]:
            if btn != sender:
                btn.setChecked(False)

        # Update mode label with selected shape
        shape_name = sender.text()
        self.mode_label.setText(f"Mode: Creating {shape_name} ROI")

        # Reset current ROI points if any
        self.roi_manager.current_roi_points = []

        # Update display
        if self.current_frame is not None:
            self.update_frame(self.current_frame)

    def toggle_sensitivity_preview(self, state):
        """Toggle sensitivity preview mode"""
        self.sensitivity_preview_mode = bool(state)
        self.update_frame(self.current_frame)

    def update_sensitivity_preview(self):
        """Update sensitivity preview when settings change"""
        if self.sensitivity_preview_mode and self.current_frame is not None:
            self.update_frame(self.current_frame)

    def update_class_combo(self):
        """Update the class combo box with current class names"""
        # Save current selection
        current_text = self.classes_combo.currentText() if self.classes_combo.count() > 0 else ""

        # Clear combo
        self.classes_combo.clear()
        self.classes_combo.addItem("All Classes")

        # Get class names
        class_dict = {}

        # Try to get class names from class manager
        if hasattr(self.roi_manager, 'class_manager') and self.roi_manager.class_manager:
            for class_info in self.roi_manager.class_manager.get_all_classes():
                class_id = class_info["class_id"]
                class_dict[class_id] = class_info["class_name"]
        else:
            # Fallback to detector's class names
            class_dict = YOLODetector.get_class_names()

        # Add all classes, sorted by ID
        for class_id, class_name in sorted(class_dict.items()):
            self.classes_combo.addItem(f"{class_id}: {class_name}")

        # Restore selection if possible
        if current_text:
            index = self.classes_combo.findText(current_text)
            if index >= 0:
                self.classes_combo.setCurrentIndex(index)

    def update_frame(self, frame):
        """Update the displayed frame with dynamic shape drawing"""
        if frame is None or not hasattr(frame, 'copy'):
            # Handle case where frame is not a valid image (like when it's an int)
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                # Use the existing frame if we have one
                display_frame = self.current_frame.copy()
            else:
                # Log error and return early if we can't proceed
                logger.error(f"Invalid frame type provided: {type(frame)}")
                return
        else:
            # Normal case - frame is a valid image
            self.current_frame = frame.copy()
            display_frame = frame.copy()

        # Get transparency level from slider (convert 1-10 range to 0.1-1.0)
        alpha = self.transparency_slider.value() / 10.0

        # Draw ROIs with the specified transparency
        display_frame = self.roi_manager.draw_rois(
            display_frame,
            show_labels=True,
            highlight_index=self.selected_roi_index,
            show_attributes=True,
            alpha=alpha
        )

        # Draw current ROI points or editing state
        if self.creating_roi:
            # Handle different drawing modes
            shape_type = self.get_current_shape_type()

            if shape_type == "rectangle" and hasattr(self, 'rectangle_dragging') and self.rectangle_dragging:
                # Draw rectangle in progress
                if len(self.roi_manager.current_roi_points) == 1:
                    start_point = self.roi_manager.current_roi_points[0]
                    end_point = self.temp_second_point
                    cv2.rectangle(display_frame, start_point, end_point, (0, 255, 255), 2)

                    # Draw corner points
                    for pt in [start_point, end_point]:
                        cv2.circle(display_frame, pt, 5, (0, 255, 255), -1)

            elif shape_type == "circle" and hasattr(self, 'circle_dragging') and self.circle_dragging:
                # Draw circle in progress
                if len(self.roi_manager.current_roi_points) == 1:
                    center = self.roi_manager.current_roi_points[0]
                    radius_point = self.temp_radius_point

                    # Calculate radius
                    radius = int(np.sqrt((center[0] - radius_point[0]) ** 2 +
                                         (center[1] - radius_point[1]) ** 2))

                    # Draw circle
                    cv2.circle(display_frame, center, radius, (0, 255, 255), 2)

                    # Draw center and radius points
                    cv2.circle(display_frame, center, 5, (0, 255, 255), -1)
                    cv2.circle(display_frame, radius_point, 5, (0, 255, 255), -1)

                    # Draw radius line
                    cv2.line(display_frame, center, radius_point, (0, 255, 255), 1)

            elif (shape_type == "polygon" or shape_type == "bezier") and self.roi_manager.current_roi_points:
                # Draw polygon or Bezier curve in progress
                points = np.array(self.roi_manager.current_roi_points, np.int32)
                cv2.polylines(display_frame, [points.reshape((-1, 1, 2))],
                              False, (0, 255, 255), 2)

                # Draw points
                for pt in self.roi_manager.current_roi_points:
                    cv2.circle(display_frame, pt, 5, (0, 255, 255), -1)

            elif shape_type == "freehand" and hasattr(self, 'freehand_drawing') and self.freehand_drawing:
                # Draw freehand path in progress
                if len(self.roi_manager.current_roi_points) > 1:
                    points = np.array(self.roi_manager.current_roi_points, np.int32)
                    cv2.polylines(display_frame, [points.reshape((-1, 1, 2))],
                                  False, (0, 255, 255), 2)

                    # Draw current point
                    current_pt = self.roi_manager.current_roi_points[-1]
                    cv2.circle(display_frame, current_pt, 5, (0, 255, 255), -1)

        elif self.editing_roi and self.selected_roi_index is not None:
            # Show sensitivity preview if enabled and we have a detector
            if (self.sensitivity_preview_mode and
                    hasattr(self, 'video_source') and
                    hasattr(self.video_source, 'detector') and
                    self.video_source.detector is not None):

                roi = self.roi_manager.rois[self.selected_roi_index]
                threshold_range = (self.threshold_min.value(), self.threshold_max.value())

                # Use detector to visualize sensitivity
                display_frame = self.video_source.detector.visualize_sensitivity(
                    display_frame, roi, threshold_range)
            else:
                # Regular ROI visualization
                # Draw all ROIs with the selected one highlighted
                display_frame = self.roi_manager.draw_rois(
                    display_frame,
                    show_labels=True,
                    highlight_index=self.selected_roi_index,
                    show_attributes=True
                )

                # Draw handles for the selected ROI
                roi = self.roi_manager.rois[self.selected_roi_index]

                if roi.roi_type == ROI.TYPE_POLYGON or roi.roi_type == ROI.TYPE_FREEHAND:
                    for i, pt in enumerate(roi.points):
                        # Highlight selected point
                        color = (0, 0, 255) if i == self.selected_point_index else (255, 255, 0)
                        cv2.circle(display_frame, pt, 8, color, -1)

                elif roi.roi_type == ROI.TYPE_RECTANGLE and len(roi.points) == 2:
                    # Draw handles at corners
                    for pt in roi.points:
                        cv2.circle(display_frame, pt, 8, (255, 255, 0), -1)

                    # Draw handles at midpoints of edges
                    x1, y1 = roi.points[0]
                    x2, y2 = roi.points[1]
                    mid_points = [
                        (x1, (y1 + y2) // 2),  # Left middle
                        (x2, (y1 + y2) // 2),  # Right middle
                        ((x1 + x2) // 2, y1),  # Top middle
                        ((x1 + x2) // 2, y2)  # Bottom middle
                    ]
                    for pt in mid_points:
                        cv2.circle(display_frame, pt, 6, (0, 255, 255), -1)

                elif roi.roi_type == ROI.TYPE_CIRCLE:
                    # Draw center and radius handles
                    if roi.center:
                        cv2.circle(display_frame, roi.center, 8, (255, 255, 0), -1)

                        # Draw radius handle
                        if roi.radius > 0:
                            radius_pt = (
                                roi.center[0] + roi.radius,
                                roi.center[1]
                            )
                            cv2.circle(display_frame, radius_pt, 8, (0, 255, 255), -1)
                            cv2.line(display_frame, roi.center, radius_pt, (0, 255, 255), 2)

                elif roi.roi_type == ROI.TYPE_BEZIER:
                    # Draw control points
                    if roi.control_points:
                        for i, pt in enumerate(roi.control_points):
                            cv2.circle(display_frame, pt, 8, (0, 255, 255), -1)
                            # Draw line connecting control points
                            if i > 0:
                                prev_pt = roi.control_points[i - 1]
                                cv2.line(display_frame, prev_pt, pt, (0, 255, 255), 1)
        else:
            # Draw all ROIs normally
            display_frame = self.roi_manager.draw_rois(display_frame)

        # Add resolution indicator
        if self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
            zoom_txt = f"{int(self.zoom_factor * 100)}%" if hasattr(self, 'zoom_factor') else "100%"
            res_text = f"Resolution: {width}x{height} | Zoom: {zoom_txt}"
            cv2.putText(display_frame, res_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        self.display_frame(display_frame)

    def display_frame(self, frame):
        """
        Convert and display a frame in the camera view

        Args:
            frame: The frame to display
        """
        # Convert the frame to QImage
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_image = QImage(frame.data, width, height,
                         bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Create a pixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)

        # Get current size of the label
        label_size = self.camera_view.size()

        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Set the pixmap to the label
        self.camera_view.setPixmap(scaled_pixmap)

    def refresh_roi_list(self):
        """Update the ROI list widget"""
        self.roi_list.clear()

        for i, roi in enumerate(self.roi_manager.rois):
            item = QListWidgetItem(f"{i + 1}: {roi.name}")
            # Set background color similar to ROI color
            color = QColor(*roi.color)
            # Make it lighter for visibility
            color.setAlpha(100)
            item.setBackground(color)
            self.roi_list.addItem(item)

    def start_roi_creation(self):
        """Start creating a new ROI with proper initialization for each shape type"""
        self.creating_roi = True
        self.editing_roi = False
        self.roi_manager.current_roi_points = []

        # Reset shape-specific variables
        self.rectangle_dragging = False
        self.circle_dragging = False
        self.freehand_drawing = False
        self.temp_second_point = None
        self.temp_radius_point = None

        # Update mode label
        shape_type = self.get_current_shape_type()
        self.mode_label.setText(f"Mode: Creating {shape_type.capitalize()} ROI")

    def start_edit_mode(self):
        """Start ROI edit mode"""
        self.editing_roi = True
        self.creating_roi = False
        self.mode_label.setText("Mode: Editing ROIs")

    def stop_edit_mode(self):
        """Stop ROI edit mode"""
        self.editing_roi = False
        self.creating_roi = False
        self.selected_roi_index = None
        self.selected_point_index = None
        self.mode_label.setText("Mode: Viewing")

    def on_roi_selected(self, row):
        """
        Handle ROI list selection

        Args:
            row: Selected row index
        """
        if row < 0 or row >= len(self.roi_manager.rois):
            self.selected_roi_index = None
            return

        self.selected_roi_index = row
        roi = self.roi_manager.rois[row]

        # Update properties panel
        self.roi_name_edit.setText(roi.name)
        self.roi_threshold_spin.setValue(roi.threshold)
        self.roi_cooldown_spin.setValue(roi.cooldown)

        # Update color button
        r, g, b = roi.color
        self.roi_color_button.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

        # Update classes list
        self.selected_classes_list.clear()
        if roi.classes_of_interest is not None:
            self.use_global_classes.setChecked(False)
            self.selected_classes_list.setEnabled(True)
            self.classes_combo.setEnabled(True)
            self.add_class_button.setEnabled(True)

            # Add classes to list
            class_dict = YOLODetector.get_class_names()
            for class_id in roi.classes_of_interest:
                class_name = class_dict.get(class_id, f"Unknown-{class_id}")
                self.selected_classes_list.addItem(f"{class_id}: {class_name}")
        else:
            self.use_global_classes.setChecked(True)

    def edit_selected_roi(self):
        """Start editing the selected ROI"""
        if self.selected_roi_index is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to edit.")
            return

        self.editing_roi = True
        self.creating_roi = False
        self.mode_label.setText(f"Mode: Editing ROI {self.selected_roi_index + 1}")

    def delete_selected_roi(self):
        """Delete the selected ROI"""
        if self.selected_roi_index is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to delete.")
            return

        reply = QMessageBox.question(self, "Delete ROI",
                                     f"Are you sure you want to delete ROI {self.selected_roi_index + 1}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.roi_manager.remove_roi(self.selected_roi_index)
            self.refresh_roi_list()
            self.selected_roi_index = None
            self.selected_point_index = None
            self.rois_changed.emit()

    def clear_all_rois(self):
        """Clear all ROIs"""
        if not self.roi_manager.rois:
            return

        reply = QMessageBox.question(self, "Clear All ROIs",
                                     "Are you sure you want to clear all ROIs?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.roi_manager.clear_all_rois()
            self.refresh_roi_list()
            self.selected_roi_index = None
            self.selected_point_index = None
            self.rois_changed.emit()

    def save_rois(self):
        """Save ROI configuration to file"""
        success = self.roi_manager.save_config()
        if success:
            QMessageBox.information(self, "ROI Configuration", "ROI configuration saved successfully.")
        else:
            QMessageBox.warning(self, "ROI Configuration", "Failed to save ROI configuration.")

    def apply_roi_properties(self):
        """Apply ROI properties from the form"""
        if self.selected_roi_index is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to apply properties.")
            return

        roi = self.roi_manager.rois[self.selected_roi_index]

        # Update ROI properties
        roi.name = self.roi_name_edit.text()
        roi.threshold = self.roi_threshold_spin.value()
        roi.cooldown = self.roi_cooldown_spin.value()

        # Update classes of interest
        if self.use_global_classes.isChecked():
            roi.classes_of_interest = None
        else:
            roi.classes_of_interest = []
            for i in range(self.selected_classes_list.count()):
                item_text = self.selected_classes_list.item(i).text()
                class_id = int(item_text.split(":")[0])
                roi.classes_of_interest.append(class_id)

        # Refresh the list to update display
        self.refresh_roi_list()
        self.rois_changed.emit()

        QMessageBox.information(self, "ROI Properties", f"Properties for ROI {roi.name} applied successfully.")

    def select_roi_color(self):
        """Open color selection dialog"""
        if self.selected_roi_index is None:
            return

        roi = self.roi_manager.rois[self.selected_roi_index]
        current_color = QColor(*roi.color)

        color = QColorDialog.getColor(current_color, self, "Select ROI Color")
        if color.isValid():
            roi.color = (color.red(), color.green(), color.blue())
            self.roi_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.refresh_roi_list()

    def toggle_classes_filter(self, state):
        """Toggle between global and custom classes for ROI"""
        is_custom = not bool(state)
        self.selected_classes_list.setEnabled(is_custom)
        self.classes_combo.setEnabled(is_custom)
        self.add_class_button.setEnabled(is_custom)

    def add_roi_class(self):
        """Add a class to the ROI's classes of interest"""
        if self.classes_combo.currentIndex() == 0:
            # "All Classes" selected
            return

        # Get selected class
        class_text = self.classes_combo.currentText()

        # Check if already in list
        for i in range(self.selected_classes_list.count()):
            if self.selected_classes_list.item(i).text() == class_text:
                return

        # Add to list
        self.selected_classes_list.addItem(class_text)

    def on_mouse_press(self, event):
        """Handle mouse press events with proper shape handling"""
        if self.current_frame is None:
            return

        # Calculate position in image considering zoom and pan
        x, y = self.get_image_position(event)

        # Apply grid snapping if enabled
        if hasattr(self, 'enable_snap') and self.enable_snap.isChecked():
            x, y = self.snap_to_grid(x, y)

        if hasattr(self, 'pan_mode_button') and self.pan_mode_button.isChecked():
            # Pan mode
            self.panning = True
            self.pan_start = (event.x(), event.y())
        elif event.button() == Qt.LeftButton:
            # Handle left-click for ROI creation
            if self.creating_roi:
                # Get current shape type
                shape_type = self.get_current_shape_type()

                if shape_type == "polygon":
                    # Add point to polygon
                    self.roi_manager.current_roi_points.append((x, y))
                elif shape_type == "rectangle":
                    if len(self.roi_manager.current_roi_points) == 0:
                        # First corner point
                        self.roi_manager.current_roi_points.append((x, y))
                        # Start rectangle dragging mode
                        self.rectangle_dragging = True
                        self.temp_second_point = (x, y)  # Initialize with same point
                    elif len(self.roi_manager.current_roi_points) == 1 and self.rectangle_dragging:
                        # Complete the rectangle with second corner
                        self.roi_manager.current_roi_points.append((x, y))
                        self.rectangle_dragging = False
                        # Show dialog to complete the ROI
                        self.complete_roi()
                elif shape_type == "circle":
                    if len(self.roi_manager.current_roi_points) == 0:
                        # Center point
                        self.roi_manager.current_roi_points.append((x, y))
                        # Start circle dragging mode
                        self.circle_dragging = True
                        self.temp_radius_point = (x, y)  # Initialize with center
                    elif len(self.roi_manager.current_roi_points) == 1 and self.circle_dragging:
                        # Complete the circle with radius point
                        self.roi_manager.current_roi_points.append((x, y))
                        self.circle_dragging = False
                        # Show dialog to complete the ROI
                        self.complete_roi()
                elif shape_type == "freehand":
                    # Add point and continue collecting points while mouse is down
                    if not hasattr(self, 'freehand_drawing') or not self.freehand_drawing:
                        self.freehand_drawing = True
                        self.roi_manager.current_roi_points = []  # Clear any existing points

                    self.roi_manager.current_roi_points.append((x, y))
                elif shape_type == "bezier":
                    # Add control point
                    self.roi_manager.current_roi_points.append((x, y))

            # Handle editing mode
            elif self.editing_roi and self.selected_roi_index is not None:
                # [existing editing code]
                pass

        elif event.button() == Qt.RightButton:
            # Right-click to complete the current shape
            if self.creating_roi:
                shape_type = self.get_current_shape_type()

                if shape_type == "rectangle" and self.rectangle_dragging:
                    # For rectangle, add the current temporary point and complete
                    if len(self.roi_manager.current_roi_points) == 1 and hasattr(self, 'temp_second_point'):
                        self.roi_manager.current_roi_points.append(self.temp_second_point)
                        self.rectangle_dragging = False
                        # Show dialog to complete the ROI
                        self.complete_roi()
                    else:
                        # Cancel if not enough points
                        self.roi_manager.current_roi_points = []
                        self.rectangle_dragging = False

                elif shape_type == "circle" and self.circle_dragging:
                    # For circle, add the current radius point and complete
                    if len(self.roi_manager.current_roi_points) == 1 and hasattr(self, 'temp_radius_point'):
                        self.roi_manager.current_roi_points.append(self.temp_radius_point)
                        self.circle_dragging = False
                        # Show dialog to complete the ROI
                        self.complete_roi()
                    else:
                        # Cancel if not enough points
                        self.roi_manager.current_roi_points = []
                        self.circle_dragging = False

                elif shape_type == "freehand" and self.freehand_drawing:
                    # Complete freehand drawing if we have enough points
                    self.freehand_drawing = False
                    if len(self.roi_manager.current_roi_points) >= 3:
                        # Complete the shape
                        self.complete_roi()
                    else:
                        # Not enough points, cancel
                        self.roi_manager.current_roi_points = []

                elif (shape_type == "polygon" or shape_type == "bezier") and len(
                        self.roi_manager.current_roi_points) >= 3:
                    # Complete polygon or bezier if enough points
                    self.complete_roi()
                else:
                    # Not enough points for a valid shape
                    if len(self.roi_manager.current_roi_points) < 3:
                        QMessageBox.warning(self, "ROI Creation",
                                            f"{shape_type.capitalize()} requires at least 3 points.")
                    self.roi_manager.current_roi_points = []
                    self.creating_roi = False
                    self.mode_label.setText("Mode: Viewing")

        # Update the display
        self.update_frame(self.current_frame)

    def on_mouse_move(self, event):
        """Handle mouse move events with dynamic shape drawing"""
        if self.current_frame is None:
            return

        # Calculate position in image
        x, y = self.get_image_position(event)

        # Apply grid snapping if enabled
        if hasattr(self, 'enable_snap') and self.enable_snap.isChecked():
            x, y = self.snap_to_grid(x, y)

        needs_update = False

        if hasattr(self, 'panning') and self.panning and hasattr(self, 'pan_start') and self.pan_start:
            # Handle panning
            # [existing panning code]
            needs_update = True
        elif self.creating_roi:
            # Handle dynamic shape drawing
            shape_type = self.get_current_shape_type()

            if shape_type == "rectangle" and self.rectangle_dragging:
                # Update second corner of rectangle
                self.temp_second_point = (x, y)
                needs_update = True
            elif shape_type == "circle" and self.circle_dragging:
                # Update radius point of circle
                self.temp_radius_point = (x, y)
                needs_update = True
            elif shape_type == "freehand" and self.freehand_drawing:
                # Add point to freehand path if enough distance from last point
                if len(self.roi_manager.current_roi_points) > 0:
                    last_point = self.roi_manager.current_roi_points[-1]
                    dist = np.sqrt((x - last_point[0]) ** 2 + (y - last_point[1]) ** 2)
                    if dist > 5:  # Minimum distance between points (5px)
                        self.roi_manager.current_roi_points.append((x, y))
                        needs_update = True
        elif self.dragging and self.selected_roi_index is not None and self.selected_point_index is not None:
            needs_update = True
        elif self.resizing and self.selected_roi_index is not None:
            # Resize the ROI
            roi = self.roi_manager.rois[self.selected_roi_index]

            # Make sure resizing_center is initialized
            if self.resizing_center is None:
                self.resizing_center = roi.get_center()

            # Only proceed if we have a valid center and start point
            if self.resizing_center is not None and self.resizing_start_point is not None:
                current_dist = np.linalg.norm(np.array((x, y)) - np.array(self.resizing_center))
                start_dist = np.linalg.norm(np.array(self.resizing_start_point) - np.array(self.resizing_center))

                # Calculate scale factor (avoid division by zero)
                if start_dist > 0:
                    scale = current_dist / start_dist

                    # Apply scaling to all points
                    new_points = []
                    for pt in self.resizing_original_points:
                        # Get vector from center to point
                        vector = np.array(pt) - np.array(self.resizing_center)
                        # Scale vector
                        scaled_vector = vector * scale
                        # Calculate new point
                        new_pt = np.array(self.resizing_center) + scaled_vector
                        new_points.append((int(new_pt[0]), int(new_pt[1])))

                    roi.points = new_points
                    needs_update = True

        # Update the display if needed
        if needs_update:
            self.update_frame(self.current_frame)

    def on_mouse_release(self, event):
        """Handle mouse release events"""
        # For freehand drawing, when mouse released, keep the points but require right-click to complete
        if event.button() == Qt.LeftButton and self.freehand_drawing:
            # Don't complete the ROI yet, just keep the points
            # User must right-click to complete
            pass

        # For other dragging operations
        if self.dragging or self.resizing:
            self.dragging = False
            self.resizing = False
            self.resizing_start_point = None
            self.resizing_original_points = None
            self.selected_point_index = None

            # Emit signal that ROIs changed
            self.rois_changed.emit()

    def complete_roi(self):
        """Complete the current ROI creation with enhanced options"""
        if not self.creating_roi:
            return

        shape_type = self.get_current_shape_type()

        # Check if we have enough points for the shape type
        if shape_type == "polygon" or shape_type == "bezier":
            if len(self.roi_manager.current_roi_points) < 3:
                QMessageBox.warning(self, "ROI Creation", "Polygon/Bezier requires at least 3 points.")
                return
        elif shape_type == "rectangle":
            if len(self.roi_manager.current_roi_points) != 2:
                QMessageBox.warning(self, "ROI Creation", "Rectangle requires exactly 2 corner points.")
                return
        elif shape_type == "circle":
            if len(self.roi_manager.current_roi_points) < 2:
                QMessageBox.warning(self, "ROI Creation", "Circle requires center and radius points.")
                return
        elif shape_type == "freehand":
            if len(self.roi_manager.current_roi_points) < 3:
                QMessageBox.warning(self, "ROI Creation", "Freehand drawing requires at least 3 points.")
                return

        # Create a dialog to get ROI properties
        dialog = QDialog(self)
        dialog.setWindowTitle("New ROI Properties")
        dialog.setMinimumWidth(400)

        dialog_layout = QFormLayout(dialog)

        # ROI name
        name_edit = QLineEdit(f"ROI {len(self.roi_manager.rois) + 1}")
        dialog_layout.addRow("Name:", name_edit)

        # Threshold
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setMinimum(0.1)
        threshold_spin.setMaximum(1.0)
        threshold_spin.setSingleStep(0.05)
        threshold_spin.setValue(0.5)
        dialog_layout.addRow("Detection Threshold:", threshold_spin)

        # Cooldown
        cooldown_spin = QSpinBox()
        cooldown_spin.setMinimum(1)
        cooldown_spin.setMaximum(3600)
        cooldown_spin.setSingleStep(5)
        cooldown_spin.setValue(60)
        cooldown_spin.setSuffix(" seconds")
        dialog_layout.addRow("Alert Cooldown:", cooldown_spin)

        # ROI Type (read-only display)
        roi_type_text = shape_type.capitalize()
        type_label = QLabel(roi_type_text)
        dialog_layout.addRow("ROI Type:", type_label)

        # Color
        color_button = QPushButton()
        color_button.setMinimumHeight(30)
        color = QColor(255, 0, 0)
        color_button.setStyleSheet(f"background-color: {color.name()};")

        def select_color():
            nonlocal color
            new_color = QColorDialog.getColor(color, dialog, "Select ROI Color")
            if new_color.isValid():
                color = new_color
                color_button.setStyleSheet(f"background-color: {color.name()};")

        color_button.clicked.connect(select_color)
        dialog_layout.addRow("Color:", color_button)

        # Classes of interest section
        classes_group = QGroupBox("Classes of Interest")
        classes_group.setCheckable(True)
        classes_group.setChecked(False)
        classes_layout = QVBoxLayout(classes_group)

        class_list = QListWidget()
        class_list.setSelectionMode(QListWidget.MultiSelection)

        # Get class names from class manager if available
        if hasattr(self, 'roi_manager') and hasattr(self.roi_manager, 'class_manager'):
            class_manager = self.roi_manager.class_manager
            for class_info in class_manager.get_all_classes():
                class_id = class_info["class_id"]
                class_name = class_info["class_name"]
                item = QListWidgetItem(f"{class_id}: {class_name}")
                item.setData(Qt.UserRole, class_id)
                class_list.addItem(item)
        else:
            # Fallback to detector's class names
            from core.detector import YOLODetector
            class_dict = YOLODetector.get_class_names()
            for class_id, class_name in sorted(class_dict.items()):
                item = QListWidgetItem(f"{class_id}: {class_name}")
                item.setData(Qt.UserRole, class_id)
                class_list.addItem(item)

        classes_layout.addWidget(class_list)

        # Class selection buttons
        class_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(class_list.selectAll)
        class_buttons.addWidget(select_all_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(class_list.clearSelection)
        class_buttons.addWidget(clear_all_btn)

        classes_layout.addLayout(class_buttons)
        dialog_layout.addRow("", classes_group)

        # Buttons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("Create ROI")
        ok_button.clicked.connect(dialog.accept)
        buttons_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_button)

        dialog_layout.addRow("", buttons_layout)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Process ROI data based on type
            roi_type = ROI.TYPE_POLYGON
            points = self.roi_manager.current_roi_points.copy()
            center = None
            radius = 0
            control_points = None

            if shape_type == "polygon":
                roi_type = ROI.TYPE_POLYGON
            elif shape_type == "rectangle":
                roi_type = ROI.TYPE_RECTANGLE
            elif shape_type == "circle":
                roi_type = ROI.TYPE_CIRCLE
                # Calculate center and radius
                if len(points) >= 2:
                    center = points[0]
                    radius_point = points[1]
                    radius = int(np.sqrt((center[0] - radius_point[0]) ** 2 +
                                         (center[1] - radius_point[1]) ** 2))
            elif shape_type == "freehand":
                roi_type = ROI.TYPE_FREEHAND
            elif shape_type == "bezier":
                roi_type = ROI.TYPE_BEZIER
                control_points = points.copy()

            # Get selected classes if enabled
            classes_of_interest = None
            if classes_group.isChecked():
                classes_of_interest = []
                for idx in range(class_list.count()):
                    item = class_list.item(idx)
                    if item.isSelected():
                        class_id = item.data(Qt.UserRole)
                        classes_of_interest.append(class_id)

            # Create ROI with enhanced properties
            roi = ROI(
                name=name_edit.text(),
                points=points,
                threshold=threshold_spin.value(),
                cooldown=cooldown_spin.value(),
                color=(color.red(), color.green(), color.blue()),
                classes_of_interest=classes_of_interest,
                roi_type=roi_type,
                center=center,
                radius=radius,
                control_points=control_points
            )

            # Add to manager
            self.roi_manager.add_roi(roi)

            # Reset current points
            self.roi_manager.current_roi_points = []

            # Reset state
            self.creating_roi = False
            self.rectangle_dragging = False
            self.circle_dragging = False
            self.freehand_drawing = False

            # Update UI
            self.refresh_roi_list()
            self.mode_label.setText("Mode: Viewing")

            # Emit signal that ROIs changed
            self.rois_changed.emit()
        else:
            # Just reset current points
            self.roi_manager.current_roi_points = []
            self.creating_roi = False
            self.rectangle_dragging = False
            self.circle_dragging = False
            self.freehand_drawing = False
            self.mode_label.setText("Mode: Viewing")

    def cancel_roi_edit(self):
        """Cancel the current ROI creation or editing"""
        if self.creating_roi:
            self.roi_manager.current_roi_points = []
            self.creating_roi = False

        if self.editing_roi:
            self.editing_roi = False
            self.selected_point_index = None

        self.mode_label.setText("Mode: Viewing")

    def get_image_position(self, event):
        """
        Convert mouse position to image coordinates with zoom and resolution support

        Args:
            event: Mouse event

        Returns:
            Tuple of (x, y) coordinates in the original image
        """
        if self.current_frame is None:
            return (0, 0)

        # Get image dimensions
        height, width = self.current_frame.shape[:2]

        # Get label dimensions
        label_width = self.camera_view.width()
        label_height = self.camera_view.height()

        # Get pixmap dimensions
        pixmap = self.camera_view.pixmap()
        if pixmap is None:
            return (0, 0)

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Calculate margins
        margin_x = (label_width - pixmap_width) // 2
        margin_y = (label_height - pixmap_height) // 2

        # Adjust for margin
        pos_x = event.x() - margin_x
        pos_y = event.y() - margin_y

        # Check if click is within image area
        if pos_x < 0 or pos_x >= pixmap_width or pos_y < 0 or pos_y >= pixmap_height:
            return (0, 0)

        # Convert to original image coordinates with zoom and pan
        if self.zoom_factor > 1.0:
            # Calculate coordinates in the zoomed view
            offset_x, offset_y = self.pan_offset

            # Get dimensions of visible region
            visible_width = width / self.zoom_factor
            visible_height = height / self.zoom_factor

            # Calculate center of visible region
            if self.zoom_center is None:
                center_x, center_y = width // 2, height // 2
            else:
                center_x, center_y = self.zoom_center

            # Adjust center with pan offset
            center_x += offset_x
            center_y += offset_y

            # Calculate boundaries of visible region
            left = center_x - (visible_width / 2)
            top = center_y - (visible_height / 2)

            # Convert click position to original image coordinates
            original_x = left + (pos_x / pixmap_width) * visible_width
            original_y = top + (pos_y / pixmap_height) * visible_height
        else:
            # Simple scaling for non-zoomed view
            original_x = (pos_x / pixmap_width) * width
            original_y = (pos_y / pixmap_height) * height

        # Clamp to image bounds
        original_x = max(0, min(width - 1, int(original_x)))
        original_y = max(0, min(height - 1, int(original_y)))

        return (original_x, original_y)

    def zoom_in(self):
        """Increase zoom factor"""
        self.zoom_factor = min(5.0, self.zoom_factor + 0.2)
        self.zoom_level.setText(f"{int(self.zoom_factor * 100)}%")
        self.update_frame(self.current_frame)

    def zoom_out(self):
        """Decrease zoom factor"""
        self.zoom_factor = max(1.0, self.zoom_factor - 0.2)
        self.zoom_level.setText(f"{int(self.zoom_factor * 100)}%")
        self.update_frame(self.current_frame)

    def zoom_reset(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.zoom_center = None
        self.pan_offset = (0, 0)
        self.zoom_level.setText("100%")
        self.update_frame(self.current_frame)

    def apply_zoom(self, frame):
        """Apply zoom to frame"""
        if self.zoom_factor == 1.0 and self.pan_offset == (0, 0):
            return frame.copy()

        height, width = frame.shape[:2]

        # Calculate zoom center if not set
        if self.zoom_center is None:
            self.zoom_center = (width // 2, height // 2)

        # Calculate the region to crop
        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)

        # Apply pan offset
        offset_x, offset_y = self.pan_offset

        # Calculate crop coordinates with zoom center and pan offset
        x_center = min(max(self.zoom_center[0] + offset_x, new_width // 2), width - new_width // 2)
        y_center = min(max(self.zoom_center[1] + offset_y, new_height // 2), height - new_height // 2)

        x1 = max(0, x_center - new_width // 2)
        y1 = max(0, y_center - new_height // 2)
        x2 = min(width, x1 + new_width)
        y2 = min(height, y1 + new_height)

        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def toggle_grid(self, state):
        """Toggle grid visibility"""
        self.show_grid = bool(state)
        self.update_frame(self.current_frame)

    def update_grid(self):
        """Update grid when settings change"""
        if self.show_grid and self.current_frame is not None:
            self.update_frame(self.current_frame)

    def draw_grid(self, frame):
        """Draw alignment grid on frame"""
        if not self.show_grid or frame is None:
            return frame

        output = frame.copy()
        h, w = output.shape[:2]
        grid_size = self.grid_size.value()

        # Draw vertical lines
        for x in range(0, w, grid_size):
            cv2.line(output, (x, 0), (x, h), (255, 255, 255, 64), 1)

        # Draw horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(output, (0, y), (w, y), (255, 255, 255, 64), 1)

        return output

    def draw_rulers(self, frame):
        """Draw ruler guides on frame"""
        if not self.enable_rulers.isChecked() or frame is None:
            return frame

        output = frame.copy()
        h, w = output.shape[:2]

        # Draw horizontal ruler at top
        ruler_height = 20
        ruler = np.ones((ruler_height, w, 3), dtype=np.uint8) * 32

        # Draw major ticks every 100px
        for x in range(0, w, 100):
            cv2.line(ruler, (x, 0), (x, ruler_height), (255, 255, 255), 1)
            cv2.putText(ruler, str(x), (x + 2, ruler_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw minor ticks every 50px
        for x in range(50, w, 100):
            cv2.line(ruler, (x, ruler_height // 2), (x, ruler_height), (255, 255, 255), 1)

        # Draw vertical ruler on left
        ruler_width = 30
        v_ruler = np.ones((h, ruler_width, 3), dtype=np.uint8) * 32

        # Draw major ticks every 100px
        for y in range(0, h, 100):
            cv2.line(v_ruler, (0, y), (ruler_width, y), (255, 255, 255), 1)
            cv2.putText(v_ruler, str(y), (2, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw minor ticks every 50px
        for y in range(50, h, 100):
            cv2.line(v_ruler, (ruler_width // 2, y), (ruler_width, y), (255, 255, 255), 1)

        # Composite rulers and frame
        result = output.copy()
        result[0:ruler_height, 0:w] = ruler
        result[0:h, 0:ruler_width] = v_ruler

        return result

    def snap_to_grid(self, x, y):
        """Snap coordinates to nearest grid point"""
        if not self.enable_snap.isChecked():
            return (x, y)

        grid_size = self.grid_size.value()
        snapped_x = round(x / grid_size) * grid_size
        snapped_y = round(y / grid_size) * grid_size

        return (snapped_x, snapped_y)

    def get_current_shape_type(self):
        """Get the currently selected shape type"""
        if hasattr(self, 'shape_polygon') and self.shape_polygon.isChecked():
            return "polygon"
        elif hasattr(self, 'shape_rectangle') and self.shape_rectangle.isChecked():
            return "rectangle"
        elif hasattr(self, 'shape_circle') and self.shape_circle.isChecked():
            return "circle"
        elif hasattr(self, 'shape_freehand') and self.shape_freehand.isChecked():
            return "freehand"
        elif hasattr(self, 'shape_bezier') and self.shape_bezier.isChecked():
            return "bezier"
        else:
            return "polygon"  # Default

    def get_roi_type_from_shape(self):
        """Convert UI shape type to ROI type constant"""
        shape_type = self.get_current_shape_type()

        if shape_type == "polygon":
            return "polygon"
        elif shape_type == "rectangle":
            return "rectangle"
        elif shape_type == "circle":
            return "circle"
        elif shape_type == "freehand":
            return "freehand"
        elif shape_type == "bezier":
            return "bezier"
        else:
            return "polygon"  # Default