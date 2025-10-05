# stellascript/audio/capture.py

import pyaudio
from contextlib import contextmanager

class AudioCapture:
    def __init__(self, format, channels, rate, chunk):
        self.format_str = format
        self.format = self._get_pyaudio_format(format)
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.pyaudio_instance = None
        self.stream = None

    def _get_pyaudio_format(self, format_str):
        if format_str == "paFloat32":
            return pyaudio.paFloat32
        # Add other formats if needed
        raise ValueError(f"Unsupported audio format: {format_str}")

    @contextmanager
    def audio_stream(self, callback):
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
                self.stream.stop_stream()
                self.stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            self.stream = None
            self.pyaudio_instance = None
