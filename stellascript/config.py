# stellascript/config.py

# Audio Configuration
FORMAT = "paFloat32"  # Corresponds to pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 512  # For VAD, 512 samples = 32ms at 16kHz

# Transcription Mode Buffering
TRANSCRIPTION_MAX_BUFFER_DURATION = 75.0  # 1min15s

# Subtitle Mode Buffering & VAD
SUBTITLE_MAX_BUFFER_DURATION = 5.0  # 5s for real-time response
VAD_SPEECH_THRESHOLD = 0.5
VAD_SILENCE_DURATION_S = 0.5
VAD_MIN_SPEECH_DURATION_S = 0.2

# Speaker Diarization
MAX_MERGE_GAP_S = 1.5  # Max silence between segments to merge

# File Transcription Chunking
TARGET_CHUNK_DURATION_S = 90.0
MAX_CHUNK_DURATION_S = 120.0
MIN_SILENCE_GAP_S = 0.5

# Transcription Padding
TRANSCRIPTION_PADDING_S = 1.5  # 1.5s of silence padding

# List of available Whisper models
MODELS = [
    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
    "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large",
    "distil-large-v2", "distil-medium.en", "distil-small.en"
]
