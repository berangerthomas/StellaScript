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
                try:
                    if self.stream.is_active():
                        # Utiliser stop_stream avec gestion du timeout
                        import threading
                        
                        # Pass stream object as an argument to make it explicit for Pylance
                        def force_stop(stream_to_stop):
                            try:
                                if stream_to_stop:
                                    stream_to_stop.stop_stream()
                            except Exception:
                                pass
                        
                        # Lancer l'arrêt dans un thread avec timeout
                        stop_thread = threading.Thread(target=force_stop, args=(self.stream,), daemon=True)
                        stop_thread.start()
                        stop_thread.join(timeout=0.2)  # Attendre max 200ms
                        
                        # Si le thread n'a pas fini, on continue quand même
                        if stop_thread.is_alive():
                            print("Stream stop timed out, continuing anyway...")
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