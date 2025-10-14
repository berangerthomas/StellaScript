# stellascript/cli.py

import argparse
import warnings
from . import config
from .logging_config import get_logger

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the Stellascript application.

    This function sets up an ArgumentParser to handle various command-line options
    for transcription, including language, model selection, input file,
    diarization, and audio enhancement. It also includes argument validation
    to ensure compatibility between different options.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
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
        choices=["block", "segment", "word"],
        default="block",
        help=(
            "Controls the timestamp granularity and output format. "
            "'block': For readable transcripts with timestamps for large text blocks. "
            "'segment': For subtitles with timestamps for short speech segments. "
            "'word': For detailed analysis with a timestamp for every single word."
        ),
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


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Validates the parsed command-line arguments to ensure they are consistent.

    This function checks for various invalid combinations of arguments, such as:
    - Using speaker count constraints in live mode.
    - Incompatible diarization and transcription modes.
    - Misuse of the similarity threshold with certain diarization methods.
    - Conflicting arguments for speaker count and similarity threshold.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        parser (argparse.ArgumentParser): The argument parser, used to report errors.

    Raises:
        SystemExit: If an invalid combination of arguments is found, the program
                    exits with an error message.
    """
    if (args.min_speakers is not None or args.max_speakers is not None) and not args.file:
        parser.error("--min-speakers and --max-speakers can only be used in file mode (--file).")

    if not args.file and args.mode == "block" and args.diarization == "cluster":
        parser.error(
            "In live mode, '--diarization cluster' is only compatible with '--mode segment'."
        )

    # The --threshold argument is only used for 'cluster' diarization.
    if args.diarization == "pyannote" and args.threshold != parser.get_default("threshold"):
        parser.error("--threshold cannot be used with --diarization pyannote.")

    # --min-speakers is only for pyannote
    if args.diarization == "cluster" and args.min_speakers is not None:
        parser.error("--min-speakers cannot be used with --diarization cluster.")

    # In cluster mode, threshold and max_speakers are mutually exclusive
    if (
        args.diarization == "cluster"
        and args.max_speakers is not None
        and args.threshold != parser.get_default("threshold")
    ):
        parser.error(
            "--threshold and --max-speakers cannot be used together with --diarization cluster."
        )
