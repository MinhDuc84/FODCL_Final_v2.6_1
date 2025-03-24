import cv2
import numpy as np
import time
import threading
import queue
import logging
import av
import platform
import os
from typing import Optional, Tuple, Union, Callable, List

logger = logging.getLogger("FOD.VideoSource")


class VideoSource:
    """
    Manages video sources (RTSP streams, local files, webcams) with improved stability
    """

    def __init__(self, source_url: str, camera_id: str = "main",
                 resize_width: int = 640, resize_height: int = 480,
                 buffer_size: int = 10,  # REDUCE from 30 to 10
                 auto_connect: bool = False,
                 rtsp_transport: str = "udp"):  # CHANGE default from tcp to udp
        """
        Initialize the video source with optimized parameters for low latency

        Args:
            source_url: URL of the video source (RTSP URL or file path)
            camera_id: Unique identifier for the camera
            resize_width: Width to resize frames to
            resize_height: Height to resize frames to
            buffer_size: Maximum number of frames to buffer
            auto_connect: Whether to connect automatically on init
            rtsp_transport: RTSP transport protocol ('tcp' or 'udp')
        """
        self.source_url = source_url
        self.camera_id = camera_id
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.initial_buffer_size = buffer_size
        self.buffer_size = buffer_size
        self.rtsp_transport = rtsp_transport

        # Network quality metrics
        self.connection_failures = 0
        self.frame_drop_rate = 0
        self.last_network_quality_check = time.time()
        self.network_quality_check_interval = 30  # seconds
        self.received_frames_count = 0
        self.dropped_frames_count = 0
        self.network_latency = 0  # ms
        self.last_successful_frame_time = 0

        # Status flags
        self.connection_ok = False
        self.is_running = False
        self.is_local_file = False
        self.connection_attempts = 0
        self.last_connection_time = 0
        self.connection_callback = None

        # Thread management
        self._frame_queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = None

        # Performance metrics
        self.fps = 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps_update_interval = 1.0  # Update FPS every second

        # Add new optimization parameters
        self.frame_skip = 0  # Skip frames when system is under load
        self.enable_hw_acceleration = True  # Use hardware acceleration when available

        # Dummy frame for when no camera is connected
        self._create_dummy_frame()

        # Auto-connect if requested
        if auto_connect:
            self.start()

    def _create_dummy_frame(self):
        """Create a dummy frame with connection instructions"""
        self.dummy_frame = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.dummy_frame, "No Camera Connected",
                    (int(self.resize_width / 2) - 150, int(self.resize_height / 2) - 30),
                    font, 1, (255, 255, 255), 2)
        cv2.putText(self.dummy_frame, "Click 'Connect Camera' to start",
                    (int(self.resize_width / 2) - 180, int(self.resize_height / 2) + 30),
                    font, 0.8, (200, 200, 200), 2)

    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Set callback function to be called when connection status changes"""
        self.connection_callback = callback

    def set_source_url(self, url: str):
        """Change the source URL"""
        if self.is_running:
            self.stop()

        self.source_url = url
        self.is_local_file = not url.lower().startswith("rtsp://")

    def start(self):
        """Start the video capture thread"""
        if self.is_running:
            logger.warning(f"Video source {self.camera_id} is already running")
            return False

        if not self.source_url:
            logger.warning("No source URL provided. Cannot start video source.")
            return False

        self.is_local_file = not self.source_url.lower().startswith("rtsp://")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_frames, daemon=True)
        self._thread.start()
        self.is_running = True
        logger.info(f"Started video source {self.camera_id}: {self.source_url}")
        return True

    def stop(self):
        """Stop the video capture thread"""
        if not self.is_running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_running = False
        self.connection_ok = False

        # Notify about connection change if callback exists
        if self.connection_callback:
            self.connection_callback(False)

        # Clear the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        logger.info(f"Stopped video source {self.camera_id}")

    def get_frame(self, timeout: float = 0.1) -> np.ndarray:
        """
        Get the next frame from the queue

        Args:
            timeout: Time to wait for a frame (seconds)

        Returns:
            Frame as numpy array (returns dummy frame if no frame available)
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            # Return dummy frame when no frames are available
            return self.dummy_frame.copy()

    def _notify_connection_change(self, is_connected: bool):
        """Notify about connection status change"""
        if self.connection_ok != is_connected:
            self.connection_ok = is_connected
            if self.connection_callback:
                self.connection_callback(is_connected)

    def _update_buffer_size(self):
        """
        Dynamically adjust buffer size based on network conditions
        - Increase buffer for poor network quality
        - Decrease buffer for good network quality to reduce latency
        """
        # Only adjust buffer for RTSP streams
        if self.is_local_file:
            return

        # Calculate frame drop rate
        total_frames = self.received_frames_count + self.dropped_frames_count
        if total_frames > 0:
            self.frame_drop_rate = self.dropped_frames_count / total_frames
        else:
            self.frame_drop_rate = 0

        # Adjust buffer size based on drop rate
        if self.frame_drop_rate > 0.1:  # More than 10% frame drops
            # Increase buffer to compensate for unstable network
            new_buffer_size = min(int(self.buffer_size * 1.5), 60)  # Max 60 frames (2 sec at 30fps)
            if new_buffer_size != self.buffer_size:
                logger.info(
                    f"Increasing buffer size from {self.buffer_size} to {new_buffer_size} due to drop rate: {self.frame_drop_rate:.2f}")

                # Create new queue with updated size
                old_queue = self._frame_queue
                self._frame_queue = queue.Queue(maxsize=new_buffer_size)

                # Transfer existing frames to new queue
                while not old_queue.empty():
                    try:
                        frame = old_queue.get_nowait()
                        if not self._frame_queue.full():
                            self._frame_queue.put(frame)
                    except queue.Empty:
                        break

                self.buffer_size = new_buffer_size
        elif self.frame_drop_rate < 0.01 and self.buffer_size > self.initial_buffer_size:
            # Network is stable, reduce buffer to minimize latency
            new_buffer_size = max(int(self.buffer_size * 0.8), self.initial_buffer_size)
            if new_buffer_size != self.buffer_size:
                logger.info(
                    f"Decreasing buffer size from {self.buffer_size} to {new_buffer_size} due to stable network")

                # Create new queue with updated size
                old_queue = self._frame_queue
                self._frame_queue = queue.Queue(maxsize=new_buffer_size)

                # Transfer most recent frames to new queue, discarding oldest if necessary
                frames = []
                while not old_queue.empty():
                    try:
                        frames.append(old_queue.get_nowait())
                    except queue.Empty:
                        break

                # Keep only the most recent frames that fit in the new buffer
                for frame in frames[-new_buffer_size:]:
                    self._frame_queue.put(frame)

                self.buffer_size = new_buffer_size

        # Reset counters for next period
        self.received_frames_count = 0
        self.dropped_frames_count = 0

    def _check_network_quality(self) -> float:
        """
        Check network quality and return a score from 0.0 (poor) to 1.0 (excellent)
        Used to determine when to switch transport protocols
        """
        # Calculate metrics
        frame_rate = self.fps
        drop_rate = self.frame_drop_rate
        connection_stability = max(0, 1.0 - (self.connection_failures / 10.0))

        # Calculate overall quality score (weighted average)
        score = (
                frame_rate / 30.0 * 0.3 +  # Normalized frame rate (30fps = ideal)
                (1.0 - drop_rate) * 0.4 +  # Frame drop rate (lower is better)
                connection_stability * 0.3  # Connection stability (higher is better)
        )

        # Clamp between 0 and 1
        return max(0.0, min(1.0, score))

    def _should_switch_transport(self) -> bool:
        """
        Determine if we should switch transport protocol based on connection quality
        """
        if self.connection_failures >= 3:
            logger.info(f"Multiple connection failures ({self.connection_failures}), recommending transport switch")
            return True

        quality_score = self._check_network_quality()
        if quality_score < 0.3:  # Very poor quality
            logger.info(f"Poor network quality score: {quality_score:.2f}, recommending transport switch")
            return True

        return False

    def switch_transport_protocol(self):
        """
        Switch between TCP and UDP transport protocols for RTSP
        """
        if self.is_local_file:
            logger.debug("Not switching transport for local file")
            return

        current_protocol = self.rtsp_transport

        # Toggle between TCP and UDP
        if current_protocol == "tcp":
            self.rtsp_transport = "udp"
        else:
            self.rtsp_transport = "tcp"

        logger.info(f"Switching RTSP transport from {current_protocol} to {self.rtsp_transport}")

        # Restart connection if currently running
        if self.is_running:
            self.stop()
            time.sleep(1)  # Brief delay before reconnecting
            self.start()

    def _read_frames(self):
        """Thread function to continuously read frames with improved stability and reduced latency"""
        backoff_delay = 1  # Initial reconnection delay (seconds)
        self.connection_attempts += 1
        self.last_connection_time = time.time()
        frame_time = time.time()
        frames_processed = 0
        frame_count = 0

        # Reset metrics
        self.received_frames_count = 0
        self.dropped_frames_count = 0

        while not self._stop_event.is_set():
            try:
                logger.info(f"Attempting to connect to {self.source_url} with transport {self.rtsp_transport}")

                # Set up hardware acceleration options
                options = {'rtsp_transport': self.rtsp_transport}

                if self.enable_hw_acceleration:
                    # Add hardware acceleration options based on platform
                    if platform.system() == 'Linux':
                        options['hwaccel'] = 'vaapi'
                    elif platform.system() == 'Windows':
                        options['hwaccel'] = 'dxva2'
                    elif platform.system() == 'Darwin':  # macOS
                        options['hwaccel'] = 'videotoolbox'

                # Open connection with options
                container = av.open(self.source_url, options=options)
                self._notify_connection_change(True)

                # Connection success, reset failure counter
                self.connection_failures = 0

                # Get video stream information for local files
                if self.is_local_file:
                    stream = container.streams.video[0]
                    video_fps = float(stream.average_rate) if stream.average_rate else 25.0
                    logger.info(f"Local video FPS: {video_fps}")
                else:
                    logger.info(f"Connected to RTSP stream: {self.source_url} with transport {self.rtsp_transport}")

                backoff_delay = 1  # Reset backoff on successful connection

                # Track frame timestamps to detect stalls
                last_frame_received = time.time()

                for frame in container.decode(video=0):
                    if self._stop_event.is_set():
                        break

                    # Implement frame skipping for performance
                    frame_count += 1
                    if self.frame_skip > 0 and frame_count % (self.frame_skip + 1) != 0:
                        continue

                    # Update last frame received time
                    current_time = time.time()
                    frame_interval = current_time - last_frame_received
                    last_frame_received = current_time
                    self.last_successful_frame_time = current_time

                    # Increment received frames count
                    self.received_frames_count += 1

                    # Convert to numpy array and resize
                    frame_array = frame.to_ndarray(format='bgr24')
                    if frame_array.shape[1] != self.resize_width or frame_array.shape[0] != self.resize_height:
                        frame_array = cv2.resize(frame_array, (self.resize_width, self.resize_height))

                    # Calculate FPS
                    frames_processed += 1

                    if current_time - frame_time >= self.fps_update_interval:
                        self.fps = frames_processed / (current_time - frame_time)
                        frame_time = current_time
                        frames_processed = 0

                        # Periodically check and adjust buffer size
                        if current_time - self.last_network_quality_check >= self.network_quality_check_interval:
                            self._update_buffer_size()
                            self.last_network_quality_check = current_time

                    # Add to queue, dropping oldest frame if full
                    if self._frame_queue.full():
                        try:
                            # Always discard oldest frame to maintain real-time feed
                            self._frame_queue.get_nowait()
                            self.dropped_frames_count += 1
                        except queue.Empty:
                            pass
                    self._frame_queue.put(frame_array)

                    # For local files, reduce sleep time for faster playback
                    if self.is_local_file:
                        time.sleep(0.5 / video_fps)  # Only sleep half as long

                    # Check for stalled stream (no frames for too long)
                    if not self.is_local_file and frame_interval > 5.0:  # 5 second threshold
                        logger.warning(f"Stream appears stalled - {frame_interval:.1f}s since last frame")
                        # Consider transport switch if quality is poor
                        if self._should_switch_transport():
                            logger.info("Stream stalled and quality is poor, will try different transport")
                            break  # Exit the frame loop to reconnect with new transport

                container.close()
                logger.warning(f"Video stream ended: {self.source_url}")

                # If we reach here for a local file, we've reached the end
                if self.is_local_file:
                    # For local files that reached the end, we stop
                    self._notify_connection_change(False)
                    break

                # Otherwise for RTSP, try to reconnect
                self._notify_connection_change(False)

            except Exception as e:
                self._notify_connection_change(False)
                logger.error(f"Error reading from {self.source_url}: {e}")
                self.connection_failures += 1

                # Check if we should switch transport after failures
                if not self.is_local_file and self._should_switch_transport():
                    logger.info(f"Switching transport after {self.connection_failures} failures")
                    # Toggle transport protocol
                    self.switch_transport_protocol()
                    backoff_delay = 1  # Reset backoff delay after protocol switch

            # Reconnection logic
            if not self._stop_event.is_set():
                logger.warning(f"Connection lost or failed. Reconnecting in {backoff_delay}s...")
                # Sleep with checking stop_event periodically
                for _ in range(backoff_delay):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)

                backoff_delay = min(backoff_delay * 2, 30)  # Exponential backoff up to 30s
                self.connection_attempts += 1
                self.last_connection_time = time.time()

        logger.info(f"Video capture thread for {self.camera_id} terminated")

    def get_status(self) -> dict:
        """Get the current status of the video source"""
        return {
            "camera_id": self.camera_id,
            "source_url": self.source_url,
            "connection_ok": self.connection_ok,
            "is_running": self.is_running,
            "fps": round(self.fps, 1),
            "resolution": f"{self.resize_width}x{self.resize_height}",
            "queue_size": self._frame_queue.qsize(),
            "buffer_size": self.buffer_size,
            "is_local_file": self.is_local_file,
            "connection_attempts": self.connection_attempts,
            "last_connection_time": self.last_connection_time,
            "rtsp_transport": self.rtsp_transport,
            "frame_drop_rate": round(self.frame_drop_rate, 3),
            "frame_skip": self.frame_skip,
            "network_quality": round(self._check_network_quality(), 2) if not self.is_local_file else 1.0
        }

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test connection to source without starting the thread

        Returns:
            Tuple of (success, message)
        """
        try:
            # Try to open the source with appropriate transport options
            options = {}
            if self.source_url.lower().startswith("rtsp://"):
                options = {'rtsp_transport': self.rtsp_transport}

            # Try to open the source
            container = av.open(self.source_url, options=options, timeout=5)

            # Try to get a frame
            for frame in container.decode(video=0):
                # Got a frame, connection is working
                container.close()
                return True, f"Connection successful using {self.rtsp_transport} transport"

            # If we get here, no frames were available
            container.close()
            return False, "Connected but no frames available"

        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_recommended_transport(self) -> str:
        """
        Test both UDP and TCP to determine which works better

        Returns:
            Recommended transport protocol ('tcp' or 'udp')
        """
        if self.is_local_file:
            return "tcp"  # Doesn't matter for local files

        results = []

        # Test TCP
        original_transport = self.rtsp_transport
        self.rtsp_transport = "tcp"
        tcp_success, tcp_msg = self.test_connection()

        # Test UDP
        self.rtsp_transport = "udp"
        udp_success, udp_msg = self.test_connection()

        # Restore original setting
        self.rtsp_transport = original_transport

        # Determine recommendation
        if tcp_success and not udp_success:
            return "tcp"
        elif udp_success and not tcp_success:
            return "udp"
        elif tcp_success and udp_success:
            # Both work, prefer UDP for lower latency
            return "udp"
        else:
            # Neither worked, default to TCP as it's more reliable
            return "tcp"