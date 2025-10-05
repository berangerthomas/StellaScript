# main.py

import time
import traceback
from stellascript.cli import parse_args
from stellascript.orchestrator import StellaScriptTranscription

def main():
    """Main function to run the transcription application."""
    args = parse_args()
    
    transcriber = None
    try:
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
        )

        if args.file:
            transcriber.transcribe_file(args.file)
        else:
            print("Starting live transcription from microphone...")
            transcriber.start_recording()
            print("Recording in progress... Press Ctrl+C to stop")

            while True:
                result = transcriber.get_transcription()
                if result:
                    # This part is for potential external use, not console display
                    # The orchestrator already prints the output
                    pass
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping process...")
        if transcriber:
            if not args.file:
                transcriber.stop_recording()
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
