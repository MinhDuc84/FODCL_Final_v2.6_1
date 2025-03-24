import logging
import os
import threading
import time
from typing import Optional, Any

from notifications.base import BaseNotifier

logger = logging.getLogger("FOD.Notifications.Sound")


class SoundNotifier(BaseNotifier):
    """
    Play sound alerts for detections
    """

    def __init__(self, sound_file: str, min_severity: int = 1,
                 cooldown: float = 2.0, volume: float = 1.0,
                 sound_duration: float = 1.0):
        """
        Initialize the sound notifier

        Args:
            sound_file: Path to sound file (.wav or .mp3)
            min_severity: Minimum severity level to trigger notifications (1=low, 2=medium, 3=high)
            cooldown: Minimum time between sound alerts in seconds
            volume: Volume level (0.0 to 1.0)
        """
        super().__init__(name="Sound", min_severity=min_severity)
        self.sound_file = sound_file
        self.cooldown = cooldown
        self.volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
        self.last_play_time = 0
        self._is_playing = False
        self._initialized = False
        self._player = None
        self._sound = None

        # Initialize sound player
        self._init_player()

    def _init_player(self):
        """Initialize the sound player"""
        try:
            # Try to import pygame
            import pygame

            # Initialize mixer only if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            # Load sound file
            if self.sound_file and os.path.exists(self.sound_file):
                self._player = pygame.mixer
                self._sound = pygame.mixer.Sound(self.sound_file)
                self._sound.set_volume(self.volume)
                self._initialized = True
                logger.info(f"Sound notifier initialized with file: {self.sound_file}")
            else:
                logger.error(f"Sound file not found: {self.sound_file}")
                if self.sound_file:
                    abs_path = os.path.abspath(self.sound_file)
                    logger.error(f"Absolute path: {abs_path}")
                    logger.error(f"Current directory: {os.getcwd()}")
                else:
                    logger.error("No sound file path provided")
        except ImportError:
            logger.error("pygame not installed, sound notifications disabled")
        except Exception as e:
            logger.error(f"Error initializing sound player: {e}")

    # Trong sound.py
    def _play_sound_thread(self):
        """Thread function to play sound"""
        try:
            self._is_playing = True
            logger.info("Playing sound alert now")

            # Play the sound
            self._sound.play()

            # Wait for sound to finish
            # Try to get the sound length, or use default duration if not available
            try:
                if hasattr(self._sound, 'get_length'):
                    sound_duration = self._sound.get_length()
                else:
                    sound_duration = self.sound_duration  # Use the duration provided during initialization
                logger.info(f"Sound duration: {sound_duration}s")
                time.sleep(sound_duration)  # Wait for sound to complete
            except Exception as e:
                logger.error(f"Error getting sound duration: {e}")
                time.sleep(self.sound_duration)  # Fallback to default duration

            logger.info(f"Sound played, now in cooldown period: {self.cooldown}s")

            # Reset playing state after sound is done
            self._is_playing = False
            self.last_play_time = time.time()  # Update last play time after sound finishes

            logger.info("Sound alert complete")
        except Exception as e:
            logger.error(f"Error playing sound: {e}")
            self._is_playing = False

    def send(self, alert: Any) -> bool:
        """
        Play sound alert

        Args:
            alert: The alert to trigger sound for

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized or self._sound is None:
            logger.error("Sound notifier not properly initialized")
            return False

        if alert.severity < self.min_severity:
            logger.debug(f"Alert severity {alert.severity} below minimum {self.min_severity}, skipping sound alert")
            return True  # Skip but return success

        # Check cooldown and if sound is already playing
        current_time = time.time()
        if current_time - self.last_play_time < self.cooldown or self._is_playing:
            logger.debug("Sound alert cooldown in effect or sound is already playing, skipping")
            return True  # Skip but return success

        try:
            # Start sound in a separate thread to avoid blocking
            logger.info(f"Triggering sound alert for ROI {alert.roi_name}")
            thread = threading.Thread(target=self._play_sound_thread)
            thread.daemon = True
            thread.start()

            self.last_play_time = current_time
            return True
        except Exception as e:
            logger.error(f"Error triggering sound alert: {e}")
            return False

    def set_volume(self, volume: float):
        """
        Set the sound volume

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1

        if self._initialized and hasattr(self, '_sound') and self._sound is not None:
            try:
                self._sound.set_volume(self.volume)
                logger.info(f"Sound volume set to {self.volume}")
            except Exception as e:
                logger.error(f"Error setting sound volume: {e}")

    def test_sound(self) -> bool:
        """
        Test the sound by playing it once

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized or self._sound is None:
            logger.error("Sound notifier not properly initialized")
            try:
                # Try to initialize again
                self._init_player()
                if not self._initialized or self._sound is None:
                    return False
            except Exception as e:
                logger.error(f"Error re-initializing sound player: {e}")
                return False

        try:
            logger.info(f"Testing sound alert with file: {self.sound_file}")
            thread = threading.Thread(target=self._play_sound_thread)
            thread.daemon = True
            thread.start()
            return True
        except Exception as e:
            logger.error(f"Error testing sound alert: {e}")
            return False