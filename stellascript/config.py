# stellascript/config.py

"""
Configuration settings for the Stellascript application.

This module defines various constants that control the behavior of the audio
processing, transcription, and diarization pipeline. These settings are
organized into sections for clarity and can be tuned to optimize performance
for different use cases.

Attributes:
    FORMAT (str): The audio format used for recording, corresponding to PyAudio's
                  `paFloat32`.
    CHANNELS (int): The number of audio channels (1 for mono).
    RATE (int): The sampling rate in Hz (16000 Hz is standard for speech).
    CHUNK (int): The number of samples per buffer, used for VAD processing.

    TRANSCRIPTION_MAX_BUFFER_DURATION (float): The maximum duration of the audio
                                               buffer for transcription in
                                               seconds.
    SUBTITLE_MAX_BUFFER_DURATION (float): The maximum duration of the audio
                                          buffer for subtitle generation in
                                          seconds.
    VAD_SPEECH_THRESHOLD (float): The sensitivity threshold for the Voice
                                  Activity Detection (VAD).
    VAD_SILENCE_DURATION_S (float): The duration of silence in seconds that
                                    triggers a segment split.
    VAD_MIN_SPEECH_DURATION_S (float): The minimum duration of speech in seconds
                                       to be considered a valid segment.

    SUBTITLE_MAX_LENGTH (int): The maximum number of characters per subtitle line.
    SUBTITLE_MAX_DURATION_S (float): The maximum duration of a single subtitle
                                     line in seconds.
    SUBTITLE_MAX_SILENCE_S (float): The maximum duration of silence to tolerate
                                    before creating a new subtitle line.

    MAX_MERGE_GAP_S (float): The maximum gap of silence in seconds between two
                             speech segments to be merged into one.

    TARGET_CHUNK_DURATION_S (float): The target duration for audio chunks when
                                     processing a file.
    MAX_CHUNK_DURATION_S (float): The maximum allowed duration for an audio chunk.
    MIN_SILENCE_GAP_S (float): The minimum duration of silence to be considered a
                               gap for chunking.

    TRANSCRIPTION_PADDING_S (float): The duration of silence padding added to
                                     audio segments before transcription.

    MODELS (list[str]): A list of available Whisper models for transcription.
"""

from typing import List

# Audio Configuration
FORMAT: str = "paFloat32"  # Corresponds to pyaudio.paFloat32
CHANNELS: int = 1
RATE: int = 16000
CHUNK: int = 512  # For VAD, 512 samples = 32ms at 16kHz

# Transcription Mode Buffering
TRANSCRIPTION_MAX_BUFFER_DURATION: float = 75.0  # 1min15s

# Subtitle Mode Buffering & VAD
SUBTITLE_MAX_BUFFER_DURATION: float = 15.0  # 15s for real-time response
VAD_SPEECH_THRESHOLD: float = 0.4  # Lower threshold for higher sensitivity
VAD_SILENCE_DURATION_S: float = 0.3  # Shorter silence duration to split segments
VAD_MIN_SPEECH_DURATION_S: float = 0.2

# Subtitle Generation
SUBTITLE_MAX_LENGTH: int = 80  # Max characters per subtitle line
SUBTITLE_MAX_DURATION_S: float = 15.0  # Max duration of a single subtitle line
SUBTITLE_MAX_SILENCE_S: float = 0.5  # Max silence to tolerate before creating a new line

# Speaker Diarization
MAX_MERGE_GAP_S: float = 5.0  # Max silence between segments to merge

# File Transcription Chunking
TARGET_CHUNK_DURATION_S: float = 90.0
MAX_CHUNK_DURATION_S: float = 120.0
MIN_SILENCE_GAP_S: float = 0.5

# Transcription Padding
TRANSCRIPTION_PADDING_S: float = 1.5  # 1.5s of silence padding

# List of available Whisper models
MODELS: List[str] = [
    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
    "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large",
    "distil-large-v2", "distil-medium.en", "distil-small.en"
]
