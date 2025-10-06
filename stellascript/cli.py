# stellascript/cli.py

import argparse
import warnings
from . import config

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio live from microphone or from a file."
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"],
        default="en",
        help="Language for transcription.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=config.MODELS,
        default="small",
        help="Whisper model to use.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a WAV audio file to transcribe.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for speaker identification (used with --diarization cluster).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["subtitle", "transcription"],
        default="transcription",
        help="Processing mode.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (file mode only).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (file mode only).",
    )
    parser.add_argument(
        "--diarization",
        type=str,
        choices=["pyannote", "cluster"],
        default="pyannote",
        help="Speaker diarization method.",
    )
    parser.add_argument(
        "--enhancement",
        type=str,
        choices=["none", "deepfilternet", "demucs"],
        default="none",
        help="Audio enhancement method.",
    )
    parser.add_argument(
        "--transcription-engine",
        type=str,
        choices=["auto", "faster-whisper", "transformers"],
        default="auto",
        help="Transcription engine.",
    )
    parser.add_argument(
        "--auto-engine-threshold",
        type=float,
        default=15.0,
        help="Duration threshold for auto engine selection.",
    )
    parser.add_argument(
        "--save-enhanced-audio",
        action="store_true",
        help="Save the enhanced audio to a new file.",
    )
    parser.add_argument(
        "--save-recorded-audio",
        action="store_true",
        help="Save the raw recorded audio from the microphone to a WAV file.",
    )

    args = parser.parse_args()

    validate_args(args, parser)
    return args

def validate_args(args, parser):
    """Validates parsed arguments."""
    if (args.min_speakers is not None or args.max_speakers is not None) and not args.file:
        parser.error("--min-speakers and --max-speakers can only be used in file mode (--file).")

    if not args.file and args.mode == "transcription" and args.diarization == "cluster":
        parser.error(
            "In live mode, '--diarization cluster' is only compatible with '--mode subtitle'."
        )

    # The --threshold argument is only used for 'cluster' diarization.
    # Warn the user if they provide it while using pyannote.
    if args.diarization == "pyannote" and args.threshold != parser.get_default("threshold"):
        warnings.warn(
            "Warning: --threshold is ignored when using '--diarization pyannote'."
        )
    
    if args.auto_engine_threshold != parser.get_default("auto_engine_threshold") and args.transcription_engine != "auto":
        warnings.warn(
            f"Warning: --auto-engine-threshold will be ignored because --transcription-engine is '{args.transcription_engine}'."
        )
