import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger("FOD.ModelTransitionManager")


class ModelTransitionManager:
    """
    Manages transitions between different models, handling class mappings and synchronization
    """

    def __init__(self, config_manager, class_manager, roi_manager=None, detector=None):
        """
        Initialize the model transition manager

        Args:
            config_manager: Configuration manager instance
            class_manager: Class manager instance
            roi_manager: ROI manager instance (optional)
            detector: YOLO detector instance (optional)
        """
        self.config_manager = config_manager
        self.class_manager = class_manager
        self.roi_manager = roi_manager
        self.detector = detector

        # Track current and previous model
        self.current_model = self.config_manager.get("yolo_model_path", "")
        self.previous_model = None

        # Register config listener for model changes
        self.config_manager.add_listener(self._handle_config_change)

    def _handle_config_change(self, key: str, value: Any):
        """
        Handle configuration changes

        Args:
            key: Configuration key that changed
            value: New value
        """
        if key == "yolo_model_path" and value != self.current_model:
            logger.info(f"Model changing from {self.current_model} to {value}")
            self.previous_model = self.current_model
            self.current_model = value

            # Handle the model transition
            if self.previous_model:
                self.handle_model_transition(self.previous_model, self.current_model)

    def handle_model_transition(self, old_model: str, new_model: str):
        """
        Handle transition between models, updating class mappings and synchronizing components

        Args:
            old_model: Previous model path or name
            new_model: New model path or name
        """
        # Check if auto mapping is enabled
        auto_mapping = self.config_manager.get("enable_auto_class_mapping", True)

        if auto_mapping:
            logger.info(f"Auto-generating class mappings from {old_model} to {new_model}")
            self._generate_class_mappings(old_model, new_model)

        # Update detector if available
        if self.detector is not None:
            self.update_detector(new_model)

        # Update ROIs if available
        if self.roi_manager is not None:
            self.update_rois(old_model, new_model)

    def _extract_model_name(self, model_path: str) -> str:
        """
        Extract model name from path

        Args:
            model_path: Full path to model file

        Returns:
            Model name (filename without extension)
        """
        return os.path.splitext(os.path.basename(model_path))[0]

    def _generate_class_mappings(self, old_model: str, new_model: str):
        """
        Generate mappings between classes in old and new models

        Args:
            old_model: Old model path
            new_model: New model path
        """
        # Extract model names
        old_model_name = self._extract_model_name(old_model)
        new_model_name = self._extract_model_name(new_model)

        # Get suggested mappings
        suggested_mappings = self.class_manager.mapper.suggest_mappings(old_model_name, new_model_name)

        if suggested_mappings:
            # Log mappings
            logger.info(f"Suggested mappings from {old_model_name} to {new_model_name}:")
            for source_id, target_id in suggested_mappings.items():
                # Get class names for better logging
                source_classes = self.class_manager.get_classes_by_model(old_model_name)
                target_classes = self.class_manager.get_classes_by_model(new_model_name)

                source_name = source_classes.get(source_id, {}).get("class_name", f"Unknown-{source_id}")
                target_name = target_classes.get(target_id, {}).get("class_name", f"Unknown-{target_id}")

                logger.info(f"  {source_id} ({source_name}) -> {target_id} ({target_name})")

                # Add mapping
                self.class_manager.mapper.add_mapping(old_model_name, new_model_name, source_id, target_id)

            # Save mappings
            self.class_manager.mapper.save_mappings()

    def update_detector(self, model_path: str):
        """
        Update detector with new model

        Args:
            model_path: Path to new model
        """
        try:
            logger.info(f"Updating detector with model: {model_path}")

            # Update model path
            self.detector.model_path = model_path

            # Load the model
            self.detector.load_model()

            # Update class definitions in class manager
            if hasattr(self.detector, 'model') and hasattr(self.detector.model, 'names'):
                class_names = self.detector.model.names
                model_name = self._extract_model_name(model_path)

                self.class_manager.update_from_model(model_name, len(class_names), class_names)

            logger.info(f"Detector updated successfully with model: {model_path}")
        except Exception as e:
            logger.error(f"Error updating detector: {e}")

    def update_rois(self, old_model: str, new_model: str):
        """
        Update ROIs to work with new model, applying class mappings as needed

        Args:
            old_model: Old model path
            new_model: New model path
        """
        # Extract model names
        old_model_name = self._extract_model_name(old_model)
        new_model_name = self._extract_model_name(new_model)

        try:
            logger.info(f"Updating ROIs for model transition from {old_model_name} to {new_model_name}")

            # Ensure the ROI manager has access to class manager
            if self.roi_manager and not hasattr(self.roi_manager, 'class_manager'):
                self.roi_manager.set_class_manager(self.class_manager)

            # Trigger ROI update through the ROI manager
            if hasattr(self.roi_manager, '_update_rois_for_model_change'):
                self.roi_manager._update_rois_for_model_change(new_model_name)
                logger.info("ROIs updated successfully for new model")
            else:
                logger.warning("ROI manager does not have _update_rois_for_model_change method")

                # Fall back to manual update approach for backwards compatibility
                for roi in self.roi_manager.rois:
                    if roi.classes_of_interest:
                        mapped_classes = []
                        unmapped_classes = []

                        for class_id in roi.classes_of_interest:
                            mapped_id = self.class_manager.mapper.get_mapped_id(
                                old_model_name, new_model_name, class_id)

                            if mapped_id is not None and mapped_id >= 0:
                                mapped_classes.append(mapped_id)
                            else:
                                unmapped_classes.append(class_id)

                        # If we found mappings, update the classes of interest
                        if mapped_classes:
                            roi.classes_of_interest = mapped_classes + unmapped_classes
                            logger.info(f"Updated classes for ROI {roi.name}, now: {roi.classes_of_interest}")

                # Save the updated ROI configuration
                if hasattr(self.roi_manager, 'save_config'):
                    self.roi_manager.save_config()

        except Exception as e:
            logger.error(f"Error updating ROIs: {e}")

    def get_mapping_status(self, old_model: str, new_model: str) -> Dict[str, Any]:
        """
        Get status of mappings between models

        Args:
            old_model: Old model path or name
            new_model: New model path or name

        Returns:
            Dictionary with mapping status information
        """
        old_model_name = self._extract_model_name(old_model)
        new_model_name = self._extract_model_name(new_model)

        # Get all classes for both models
        old_classes = self.class_manager.get_classes_by_model(old_model_name)
        new_classes = self.class_manager.get_classes_by_model(new_model_name)

        # Get existing mappings
        mapping_key = f"{old_model_name}:{new_model_name}"
        mappings = self.class_manager.mapper.mappings.get(mapping_key, {})

        # Calculate statistics
        total_old = len(old_classes)
        total_new = len(new_classes)
        mapped_count = len(mappings)
        unmapped_count = total_old - mapped_count

        return {
            "old_model": old_model_name,
            "new_model": new_model_name,
            "total_old_classes": total_old,
            "total_new_classes": total_new,
            "mapped_classes": mapped_count,
            "unmapped_classes": unmapped_count,
            "mappings": mappings,
            "old_classes": old_classes,
            "new_classes": new_classes
        }

    def show_mapping_dialog(self, old_model: str, new_model: str, parent=None):
        """
        Show dialog for manual class mapping

        Args:
            old_model: Old model path or name
            new_model: New model path or name
            parent: Parent widget for the dialog
        """
        # This would be implemented in the UI code
        # Here we just log that it would be shown
        logger.info(f"Would show mapping dialog for {old_model} to {new_model}")

        # In real implementation, this would:
        # 1. Show a dialog with classes from both models
        # 2. Allow user to create mappings
        # 3. Save mappings to the mapper
        pass