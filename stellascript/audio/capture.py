# stellascript/audio/capture.py

"""
Handles audio capture from the microphone using PyAudio.
"""

import threading
from contextlib import contextmanager
from typing import Callable, Generator, Optional

import pyaudio

from ..logging_config import get_logger

logger = get_logger(__name__)


class AudioCapture:
    """
    A class to manage audio recording from the microphone.

    This class provides a context manager to handle the lifecycle of a PyAudio
    stream, ensuring that resources are properly opened and closed.
    """

    def __init__(self, format: str, channels: int, rate: int, chunk: int) -> None:
        """
        Initializes the AudioCapture instance.

        Args:
            format (str): The audio format string (e.g., "paFloat32").
            channels (int): The number of audio channels.
            rate (int): The sampling rate in Hz.
            chunk (int): The number of frames per buffer.
        """
        self.format_str: str = format
        self.format: int = self._get_pyaudio_format(format)
        self.channels: int = channels
        self.rate: int = rate
        self.chunk: int = chunk
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

    def _get_pyaudio_format(self, format_str: str) -> int:
        """
        Converts a format string to a PyAudio format constant.

        Args:
            format_str (str): The string representation of the format.

        Returns:
            int: The corresponding PyAudio format constant.

        Raises:
            ValueError: If the format string is not supported.
        """
        if format_str == "paFloat32":
            return pyaudio.paFloat32
        # Add other formats if needed
        raise ValueError(f"Unsupported audio format: {format_str}")

    @contextmanager
    def audio_stream(self, callback: Callable) -> Generator[Optional[pyaudio.Stream], None, None]:
        """
        A context manager for opening and managing a PyAudio stream.

        Args:
            callback (Callable): The callback function to process audio chunks.

        Yields:
            Optional[pyaudio.Stream]: The PyAudio stream object.
        """
        self.pyaudio_instance = pyaudio.PyAudio()
        try:
            self.stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=callback,
                start=False,
            )
            yield self.stream
        finally:
            if self.stream:
                try:
                    if self.stream.is_active():
                        # Use stop_stream with timeout management
                        def force_stop(stream_to_stop: pyaudio.Stream) -> None:
                            try:
                                if stream_to_stop:
                                    stream_to_stop.stop_stream()
                            except Exception:
                                pass

                        # Run the stop in a thread with a timeout
                        stop_thread = threading.Thread(target=force_stop, args=(self.stream,), daemon=True)
                        stop_thread.start()
                        stop_thread.join(timeout=0.2)  # Wait max 200ms

                        # If the thread is still running, continue anyway
                        if stop_thread.is_alive():
                            logger.warning("Stream stop timed out, continuing anyway")
                except Exception:
                    pass

                try:
                    self.stream.close()
                except Exception:
                    pass

            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                except Exception:
                    pass

            self.stream = None
            self.pyaudio_instance = None
