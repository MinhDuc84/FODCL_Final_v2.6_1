import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any, Optional
import logging
import threading

logger = logging.getLogger("FOD.Detector")


class YOLODetector:
    """
    Wrapper for YOLO model to perform object detection with small object optimizations
    """
    # Class-level dictionary for class names
    CLASS_NAMES = {
        0: "AdjustableWrench_01", 1: "AdjustableWrench", 2: "Battery",
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

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.25,
                 use_gpu: bool = True, classes_of_interest: Optional[List[int]] = None,
                 class_manager=None,
                 input_size: tuple = (640, 640),  # NEW PARAMETER
                 enable_half_precision: bool = True):  # NEW PARAMETER
        """
        Initialize YOLO detector with optimizations for small objects

        Args:
            model_path: Path to YOLO model file
            confidence: Confidence threshold for detections
            use_gpu: Whether to use GPU acceleration
            classes_of_interest: List of class IDs to detect (None = all classes)
            class_manager: Class manager instance for dynamic class mapping
            input_size: Model input size (width, height)
            enable_half_precision: Whether to use FP16 precision (faster on GPU)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.use_gpu = use_gpu
        self.classes_of_interest = classes_of_interest
        self.model = None
        self.class_manager = class_manager
        self._dynamic_class_names = {}
        self.input_size = input_size
        self.enable_half_precision = enable_half_precision

        # Detection cache to avoid redundant processing
        self._detection_cache = {}
        self._detection_cache_max_size = 10
        self._cache_lock = threading.Lock()

        # Performance metrics
        self.last_inference_time = 0
        self.inference_times = []

        # Update dynamic class names if class manager is provided
        if self.class_manager:
            self._update_dynamic_class_names()

            # Register for class changes if class manager supports it
            if hasattr(self.class_manager, "add_listener"):
                self.class_manager.add_listener(self._handle_class_change)

        self.input_size = input_size
        self.enable_half_precision = enable_half_precision

        # Performance tracking
        self.inference_times = []
        self.last_inference_time = 0

        # Load model if path provided
        if model_path:
            self.load_model()
            # Apply small object optimizations
            self.optimize_for_small_objects()

    def load_model(self):
        """Load YOLO model with appropriate device selection and optimizations"""
        if not self.model_path:
            logger.warning("No model path provided, skipping model loading")
            return

        try:
            # Load model with optimizations
            self.model = YOLO(self.model_path)

            if self.use_gpu and torch.cuda.is_available():
                self.model.to("cuda")

                # Apply half-precision (FP16) for faster inference
                if self.enable_half_precision:
                    self.model.model.half()

                logger.info("YOLO Model loaded on GPU with optimizations")
            else:
                self.model.to("cpu")
                logger.info("YOLO Model loaded on CPU")

            # Update model parameters for faster inference
            if hasattr(self.model, 'overrides'):
                # Override model parameters for optimized inference
                self.model.overrides['conf'] = self.confidence
                self.model.overrides['iou'] = 0.4  # Reduced IoU threshold
                self.model.overrides['imgsz'] = self.input_size

            # Update dynamic class names from model if available
            if hasattr(self.model, 'names'):
                model_classes = {}
                for idx, name in self.model.names.items():
                    model_classes[int(idx)] = name

                # Only update if we don't have class manager
                if not self.class_manager:
                    self._dynamic_class_names = model_classes
                    logger.info(f"Updated class names from model: {len(model_classes)} classes")

                # If we have class manager, update it with model classes
                elif hasattr(self.class_manager, "update_from_model"):
                    # Extract model name from path
                    import os
                    model_name = os.path.splitext(os.path.basename(self.model_path))[0]

                    # Update class manager
                    self.class_manager.update_from_model(model_name, len(model_classes), model_classes)
                    logger.info(f"Updated class manager with {len(model_classes)} classes from model")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def optimize_for_small_objects(self):
        """Apply optimizations for small object detection"""
        if self.model is None:
            return

        logger.info("Applying small object detection optimizations")

        # Set NMS parameters for better small object detection
        if hasattr(self.model, 'overrides'):
            # Lower IoU threshold to retain more small objects
            self.model.overrides['iou'] = 0.25

            # Increase maximum detections to catch more small objects
            self.model.overrides['max_det'] = 100

            # Use higher resolution input for small objects
            self.model.overrides['imgsz'] = self.input_size

            logger.info(f"Model optimized for small objects with input size {self.input_size}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform object detection on a frame with optimizations for small objects"""
        if self.model is None:
            logger.error("Model not loaded")
            return []

        try:
            # Calculate frame hash for caching (simple but effective strategy)
            frame_hash = hash(frame.tobytes()[:1000])  # Only hash part of the frame for efficiency

            # Check cache
            with self._cache_lock:
                if frame_hash in self._detection_cache:
                    return self._detection_cache[frame_hash].copy()

            # Track inference time
            start_time = time.time()

            # Resize frame to model input size for consistent and faster inference
            input_frame = cv2.resize(frame, self.input_size)

            # Run inference with optimized settings
            results = self.model(input_frame,
                                 conf=self.confidence,
                                 verbose=False,
                                 agnostic_nms=True,  # Class-agnostic NMS for better efficiency
                                 max_det=100)[0]  # Higher max detections for small objects

            # Update inference time metrics
            self.last_inference_time = time.time() - start_time
            self.inference_times.append(self.last_inference_time)
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]

            detections = []

            if results.boxes is not None:
                # Move tensor computation to CPU for post-processing
                boxes_data = results.boxes.data.cpu().numpy()

                # Calculate scale factors for bbox coordinate correction
                orig_h, orig_w = frame.shape[:2]
                model_h, model_w = self.input_size
                scale_x = orig_w / model_w
                scale_y = orig_h / model_h

                for box in boxes_data:
                    x1, y1, x2, y2, conf, class_id = box
                    class_id = int(class_id)

                    # Skip if not in classes of interest
                    if self.classes_of_interest is not None and class_id not in self.classes_of_interest:
                        continue

                    # Scale coordinates back to original image size
                    x1_scaled = int(x1 * scale_x)
                    y1_scaled = int(y1 * scale_y)
                    x2_scaled = int(x2 * scale_x)
                    y2_scaled = int(y2 * scale_y)

                    # Calculate center point
                    center_x = int((x1_scaled + x2_scaled) / 2)
                    center_y = int((y1_scaled + y2_scaled) / 2)

                    detection = {
                        "class_id": class_id,
                        "class_name": self.get_dynamic_class_names().get(class_id, f"Unknown-{class_id}"),
                        "confidence": float(conf),
                        "bbox": (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                        "center": (center_x, center_y)
                    }
                    detections.append(detection)

            # Cache the result
            with self._cache_lock:
                self._detection_cache[frame_hash] = detections.copy()

                # Limit cache size
                if len(self._detection_cache) > self._detection_cache_max_size:
                    # Remove oldest entry (first key)
                    self._detection_cache.pop(next(iter(self._detection_cache)))

            return detections
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def get_detection_stats(self) -> Dict[str, float]:
        """Get detection performance statistics"""
        stats = {
            "last_inference_time": self.last_inference_time * 1000,  # ms
            "avg_inference_time": 0,
            "max_inference_time": 0,
            "min_inference_time": 0
        }

        if self.inference_times:
            stats["avg_inference_time"] = sum(self.inference_times) / len(self.inference_times) * 1000
            stats["max_inference_time"] = max(self.inference_times) * 1000
            stats["min_inference_time"] = min(self.inference_times) * 1000

        return stats

    def set_class_manager(self, class_manager):
        """Set the class manager for dynamic class handling"""
        self.class_manager = class_manager
        self._update_dynamic_class_names()

        # Register for class changes
        if hasattr(self.class_manager, "add_listener"):
            self.class_manager.add_listener(self._handle_class_change)

    def _handle_class_change(self, event):
        """Handle class change events"""
        # Update dynamic class names when classes change
        if event.action in ["add", "update", "delete", "import", "model_update"]:
            self._update_dynamic_class_names()
            logger.info("Updated detector class mappings due to class changes")

    def _update_dynamic_class_names(self):
        """Update dynamic class names from class manager"""
        if not self.class_manager:
            return

        try:
            # Get class names from class manager
            self._dynamic_class_names = {}

            # Get all classes
            for class_info in self.class_manager.get_all_classes():
                class_id = class_info["class_id"]
                self._dynamic_class_names[class_id] = class_info["class_name"]

            logger.info(f"Updated dynamic class names, found {len(self._dynamic_class_names)} classes")
        except Exception as e:
            logger.error(f"Error updating dynamic class names: {e}")

    @classmethod
    def get_class_names(cls) -> Dict[int, str]:
        """Class method to get the class names dictionary without initializing a model"""
        return cls.CLASS_NAMES

    def get_dynamic_class_names(self) -> Dict[int, str]:
        """Instance method to get the dynamic class names from class manager or model"""
        # If we have dynamic class names from class manager, use those
        if self._dynamic_class_names:
            return self._dynamic_class_names

        # If model is loaded, try to get class names from model
        if self.model and hasattr(self.model, 'names'):
            return self.model.names

        # Fall back to class-level dictionary
        return self.CLASS_NAMES

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]],
                        highlight_in_roi: Optional[List[int]] = None) -> np.ndarray:
        """
        Draw detection boxes on frame

        Args:
            frame: The original frame
            detections: List of detection dictionaries
            highlight_in_roi: List of indices of detections that are inside ROIs

        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            conf = det["confidence"]

            # Use green for detections in ROI, gray for others
            color = (0, 255, 0) if highlight_in_roi is not None and i in highlight_in_roi else (200, 200, 200)

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output_frame

    def visualize_sensitivity(self, frame: np.ndarray, roi: 'ROI',
                              threshold_range: Tuple[float, float]) -> np.ndarray:
        """
        Visualize detection sensitivity within an ROI

        Args:
            frame: Input frame
            roi: ROI to visualize sensitivity for
            threshold_range: Tuple of (min_threshold, max_threshold)

        Returns:
            Frame with sensitivity visualization
        """
        if self.model is None:
            return frame

        # Extract min and max thresholds from the range tuple
        min_threshold, max_threshold = threshold_range

        # Run detector with low confidence to get potential detections
        orig_confidence = self.confidence
        self.confidence = min_threshold * 0.5  # Use half of min threshold

        # Get all possible detections
        detections = self.detect(frame)

        # Restore original confidence
        self.confidence = orig_confidence

        # Create visualization frame
        vis_frame = frame.copy()

        # Draw detections with color based on confidence
        for det in detections:
            # Skip if not in ROI
            center_point = det["center"]
            if not roi.contains_point(center_point):
                continue

            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            # Calculate color based on confidence relative to threshold range
            min_t, max_t = threshold_range
            if conf < min_t:
                # Below minimum threshold - blue with transparency
                color = (255, 0, 0)
                alpha = 0.3
            elif conf > max_t:
                # Above maximum threshold - red (solid)
                color = (0, 0, 255)
                alpha = 0.7
            else:
                # In threshold range - gradient from green to yellow to red
                ratio = (conf - min_t) / (max_t - min_t)
                if ratio < 0.5:
                    # Green to yellow
                    g = 255
                    r = int(255 * (ratio * 2))
                    b = 0
                else:
                    # Yellow to red
                    r = 255
                    g = int(255 * (1 - (ratio - 0.5) * 2))
                    b = 0
                color = (b, g, r)
                alpha = 0.5 + (ratio * 0.3)  # Transparency from 0.5 to 0.8

            # Create overlay for filled area
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Apply transparency
            cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw confidence value
            label = f"{det['class_name']}: {conf:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw threshold legend
        legend_height = 30
        legend_width = 200
        legend_x = 20
        legend_y = vis_frame.shape[0] - legend_height - 20

        # Draw gradient bar
        for i in range(legend_width):
            ratio = i / legend_width
            if ratio < 0.5:
                # Blue to green
                r = 0
                g = int(255 * (ratio * 2))
                b = int(255 * (1 - ratio * 2))
            elif ratio < 0.75:
                # Green to yellow
                g = 255
                r = int(255 * ((ratio - 0.5) * 4))
                b = 0
            else:
                # Yellow to red
                r = 255
                g = int(255 * (1 - (ratio - 0.75) * 4))
                b = 0

            color = (b, g, r)
            cv2.line(vis_frame, (legend_x + i, legend_y),
                     (legend_x + i, legend_y + legend_height), color, 1)

        # Draw threshold markers
        min_x = legend_x
        max_x = legend_x + legend_width
        y_mid = legend_y + legend_height // 2

        # Min threshold marker
        min_pos = legend_x
        cv2.drawMarker(vis_frame, (min_pos, y_mid), (255, 255, 255),
                       cv2.MARKER_TRIANGLE_DOWN, 10, 2)
        cv2.putText(vis_frame, f"Min: {min_threshold:.2f}",
                    (min_pos - 20, legend_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ROI threshold marker
        roi_pos = int(legend_x + (legend_width * ((roi.threshold - min_threshold) /
                                                  (max_threshold - min_threshold))))
        roi_pos = max(min_x, min(max_x, roi_pos))
        cv2.drawMarker(vis_frame, (roi_pos, y_mid), (0, 255, 255),
                       cv2.MARKER_TRIANGLE_DOWN, 10, 2)
        cv2.putText(vis_frame, f"ROI: {roi.threshold:.2f}",
                    (roi_pos - 20, legend_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return vis_frame