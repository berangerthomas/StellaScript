# main.py

import time
import traceback
import os
import sys
from stellascript.cli import parse_args
from stellascript.orchestrator import StellaScriptTranscription
from stellascript.logging_config import get_logger

logger = get_logger('stellascript.main')

def main():
    """Main function to run the transcription application."""
    args = parse_args()

    transcriber = None
    try:
        logger.info("Initializing StellaScript transcription")
        transcriber = StellaScriptTranscription(
            model_id=args.model,
            language=args.language,
            similarity_threshold=args.threshold,
            mode=args.mode,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            diarization_method=args.diarization,
            enhancement_method=args.enhancement,
            transcription_engine=args.transcription_engine,
            auto_engine_threshold=args.auto_engine_threshold,
            save_enhanced_audio=args.save_enhanced_audio,
            save_recorded_audio=args.save_recorded_audio,
        )

        if args.file:
            if not os.path.exists(args.file):
                logger.error(f"The specified file does not exist: {args.file}")
                sys.exit(1)
            logger.info(f"Starting file transcription: {args.file}")
            transcriber.transcribe_file(args.file)
        else:
            logger.info("Starting live transcription from microphone")
            transcriber.start_recording()
            logger.info("Recording active - Press Ctrl+C to stop")
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal - stopping process")
        if transcriber:
            transcriber.stop_recording()
            if transcriber.save_recorded_audio and not args.file:
                transcriber.save_audio()
        logger.info("Transcription stopped by user")

    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        # S'assurer de bien fermer même en cas d'erreur
        if transcriber and not args.file:
            try:
                transcriber.stop_recording()
            except:
                pass

if __name__ == "__main__":
    main()
