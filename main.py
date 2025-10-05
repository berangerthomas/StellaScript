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
            save_enhanced_audio=args.save_enhanced_audio,
            save_recorded_audio=args.save_recorded_audio,
        )

        if args.file:
            transcriber.transcribe_file(args.file)
        else:
            print("Starting live transcription from microphone...")
            transcriber.start_recording()
            print("Recording... Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping process...")
        if transcriber:
            transcriber.stop_recording()
            if transcriber.save_recorded_audio and not args.file:
                transcriber.save_audio()
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        # S'assurer de bien fermer même en cas d'erreur
        if transcriber and not args.file:
            try:
                transcriber.stop_recording()
            except:
                pass

if __name__ == "__main__":
    main()
