import logging
import platform
import time
from typing import Dict, Any
import datetime

logger = logging.getLogger("FOD.SystemInfo")


class SystemInfo:
    """
    Class for collecting system information and monitoring resources
    """

    def __init__(self):
        """Initialize the system info monitor"""
        self.start_time = time.time()
        self.last_update_time = 0
        self.info_cache = {}
        self.update_interval = 1.0  # Reduced from 2.0 to match UI update frequency
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

        # Initialize resource monitoring
        try:
            import psutil
            self.psutil_available = True
            # Initialize CPU monitoring to get first measurement
            psutil.cpu_percent(interval=0.1)
        except ImportError:
            logger.warning("psutil not installed. Limited system monitoring available.")
            self.psutil_available = False

        # Check for GPU monitoring
        try:
            import GPUtil
            self.gputil_available = True
        except ImportError:
            logger.warning("GPUtil not installed. GPU monitoring not available.")
            self.gputil_available = False

        # Get initial system info
        self.get_system_info()

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system information

        Returns:
            Dictionary with system information
        """
        # Check if we need to update the cache
        current_time = time.time()
        if (current_time - self.last_update_time < self.update_interval and 
            self.info_cache and 
            self.consecutive_errors < self.max_consecutive_errors):
            return self.info_cache

        info = {}

        # Basic system info
        info["os"] = platform.system()
        info["os_version"] = platform.version()
        info["python_version"] = platform.python_version()
        info["hostname"] = platform.node()

        # Uptime
        uptime_seconds = int(current_time - self.start_time)
        info["uptime"] = str(datetime.timedelta(seconds=uptime_seconds))

        # CPU, memory and disk info from psutil
        if self.psutil_available:
            try:
                import psutil

                # CPU info - use non-blocking call for better UI responsiveness
                info["cpu_percent"] = psutil.cpu_percent(interval=None)
                info["cpu_count"] = psutil.cpu_count(logical=True)

                # Memory info
                memory = psutil.virtual_memory()
                info["memory_total"] = self._format_bytes(memory.total)
                info["memory_available"] = self._format_bytes(memory.available)
                info["memory_used"] = self._format_bytes(memory.used)
                info["memory_percent"] = memory.percent

                # Disk info
                disk = psutil.disk_usage('/')
                info["disk_total"] = self._format_bytes(disk.total)
                info["disk_free"] = self._format_bytes(disk.free)
                info["disk_used"] = self._format_bytes(disk.used)
                info["disk_percent"] = disk.percent
                
                self.consecutive_errors = 0  # Reset error counter on success
            except Exception as e:
                logger.error(f"Error getting system info from psutil: {e}")
                self.consecutive_errors += 1
                info["cpu_percent"] = 0
                info["memory_percent"] = 0
                info["disk_percent"] = 0
        else:
            info["cpu_percent"] = 0
            info["memory_percent"] = 0
            info["disk_percent"] = 0

        # GPU info
        if self.gputil_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()

                if gpus:
                    # If multiple GPUs, calculate average load
                    gpu_count = len(gpus)
                    if gpu_count == 1:
                        # Single GPU
                        gpu = gpus[0]
                        info["gpu_name"] = gpu.name
                        info["gpu_driver"] = gpu.driver
                        info["gpu_memory_total"] = self._format_bytes(gpu.memoryTotal * 1024 * 1024)
                        info["gpu_memory_used"] = self._format_bytes(gpu.memoryUsed * 1024 * 1024)
                        info["gpu_percent"] = round(gpu.load * 100, 1)
                    else:
                        # Multiple GPUs - average the load
                        avg_load = sum(gpu.load for gpu in gpus) / gpu_count
                        total_memory = sum(gpu.memoryTotal for gpu in gpus)
                        used_memory = sum(gpu.memoryUsed for gpu in gpus)
                        
                        info["gpu_name"] = f"{gpu_count} GPUs"
                        info["gpu_memory_total"] = self._format_bytes(total_memory * 1024 * 1024)
                        info["gpu_memory_used"] = self._format_bytes(used_memory * 1024 * 1024)
                        info["gpu_percent"] = round(avg_load * 100, 1)
                else:
                    info["gpu_name"] = "No GPU detected"
                    info["gpu_percent"] = 0
                    
                self.consecutive_errors = 0  # Reset error counter on success
            except Exception as e:
                logger.error(f"Error getting GPU info: {e}")
                self.consecutive_errors += 1
                info["gpu_percent"] = 0
        else:
            info["gpu_percent"] = 0

        # Update cache
        self.info_cache = info
        self.last_update_time = current_time

        return info

    def _format_bytes(self, bytes: int) -> str:
        """
        Format bytes to human-readable string

        Args:
            bytes: Number of bytes

        Returns:
            Formatted string (e.g. "1.23 GB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024 or unit == 'TB':
                return f"{bytes:.2f} {unit}"
            bytes /= 1024

    def get_formatted_info(self) -> str:
        """
        Get formatted system information as a multiline string

        Returns:
            Formatted information string
        """
        info = self.get_system_info()

        lines = [
            f"System: {info.get('os')} {info.get('os_version')}",
            f"Hostname: {info.get('hostname')}",
            f"Python: {info.get('python_version')}",
            f"Uptime: {info.get('uptime')}",
            f"CPU Usage: {info.get('cpu_percent')}% ({info.get('cpu_count')} cores)",
            f"Memory: {info.get('memory_used')} / {info.get('memory_total')} ({info.get('memory_percent')}%)",
            f"Disk: {info.get('disk_used')} / {info.get('disk_total')} ({info.get('disk_percent')}%)"
        ]

        if 'gpu_name' in info:
            lines.append(f"GPU: {info.get('gpu_name')}")
            lines.append(f"GPU Memory: {info.get('gpu_memory_used')} / {info.get('gpu_memory_total')}")
            lines.append(f"GPU Usage: {info.get('gpu_percent')}%")

        return "\n".join(lines)