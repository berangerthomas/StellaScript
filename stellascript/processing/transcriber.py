# stellascript/processing/transcriber.py

import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
from logging import getLogger

logger = getLogger(__name__)

class Transcriber:
    def __init__(self, model_id, device, engine, auto_engine_threshold, language, mode):
        self.device = device
        self.engine = engine
        self.auto_engine_threshold = auto_engine_threshold
        self.language = language
        self.model_id = model_id
        self.mode = mode

        self.faster_whisper_model = None
        self.transformers_processor = None
        self.transformers_model = None
        self.transformers_pipeline = None # Add this line

        self._load_models()

    def _load_models(self):
        """Load transcription model(s) based on engine configuration."""
        if self.engine in ["faster-whisper", "auto"]:
            try:
                logger.info(f"Loading faster-whisper model '{self.model_id}'...")
                compute_type = "float16" if self.device.type == "cuda" else "float32"
                self.faster_whisper_model = WhisperModel(
                    self.model_id, device=self.device.type, compute_type=compute_type
                )
                logger.info(f"Faster-whisper model '{self.model_id}' loaded successfully on {self.device.type} with {compute_type} compute type.")
            except Exception:
                logger.exception(f"Failed to load faster-whisper model '{self.model_id}'.")
                if self.engine == "faster-whisper":
                    raise

        if self.engine in ["transformers", "auto"]:
            model_name = f"openai/whisper-{self.model_id}"
            try:
                logger.info(f"Loading transformers model '{model_name}'...")
                torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                
                self.transformers_pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=model_name,
                    torch_dtype=torch_dtype,
                    device=self.device,
                )
                logger.info(f"Transformers model '{model_name}' loaded successfully on {self.device.type} with {torch_dtype} dtype.")
            except Exception:
                logger.exception(f"Failed to load transformers model '{model_name}'.")
                if self.engine == "transformers":
                    raise

        if self.engine == "auto":
            logger.info(f"Transcription engine mode: 'auto' (threshold: {self.auto_engine_threshold}s)")
            logger.info(f"Segments < {self.auto_engine_threshold}s will use 'transformers', longer segments will use 'faster-whisper'.")
        else:
            logger.info(f"Transcription engine mode: '{self.engine}' (all segments).")

    def transcribe_segment(self, audio_data, rate, padding_duration, word_timestamps=False):
        """Transcribe audio segment with intelligent engine selection."""
        audio_duration = len(audio_data) / rate
        
        if audio_duration < 0.5:
            logger.debug(f"Segment too short ({audio_duration:.2f}s), skipping")
            return ""
        
        padding_samples = int(padding_duration * rate)
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        
        padded_audio = np.concatenate([silence_padding, audio_data, silence_padding])
        
        # Determine which engine to use
        use_transformers = False
        if self.engine == "transformers":
            use_transformers = True
            logger.info(f"Transcribing a {audio_duration:.2f}s segment with 'transformers' (user-selected).")
        elif self.engine == "auto":
            use_transformers = audio_duration < self.auto_engine_threshold
            engine_name = "transformers" if use_transformers else "faster-whisper"
            logger.info(f"Transcribing a {audio_duration:.2f}s segment with '{engine_name}' (auto-selected).")
        else: # faster-whisper
            use_transformers = False
            logger.info(f"Transcribing a {audio_duration:.2f}s segment with 'faster-whisper' (user-selected).")

        # In subtitle mode, word timestamps are required.
        # If the selected engine doesn't support them, we might need a fallback or warning.
        # For now, both engines are being adapted to support it.
        if self.mode == "subtitle" and not word_timestamps:
            logger.debug("Subtitle mode implies word_timestamps=True.")
            word_timestamps = True

        if use_transformers:
            return self._transcribe_with_transformers(padded_audio, rate, word_timestamps=word_timestamps)
        else:
            return self._transcribe_with_faster_whisper(padded_audio, word_timestamps=word_timestamps)

    def _transcribe_with_faster_whisper(self, audio_data, word_timestamps=False):
        """Transcribe with faster-whisper."""
        if self.faster_whisper_model is None:
            logger.warning("faster-whisper model not loaded. Returning empty transcription.")
            return "" if not word_timestamps else ([], "")

        segments, info = self.faster_whisper_model.transcribe(
            audio_data,
            language=self.language,
            task="transcribe",
            beam_size=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt="Transcription précise et naturelle sans répétitions.",
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=[-1],
            without_timestamps=False,
            max_initial_timestamp=len(audio_data) / 16000.0,
            word_timestamps=word_timestamps,
            prepend_punctuations="\"'¿([{-",
            append_punctuations="\"'.。,，!！?？:：)]}、",
        )
        
        # Convert generator to a list to allow multiple iterations.
        segments_list = list(segments)
        text = "".join(segment.text for segment in segments_list)

        # Check for repetitive transcriptions by analyzing unique word ratio.
        words = text.split()
        if len(words) > 10:
            unique_word_ratio = len(set(words)) / len(words)
            if unique_word_ratio < 0.3:
                logger.warning(
                    f"Low unique word ratio detected ({unique_word_ratio:.2f}). "
                    "The transcription may be repetitive."
                )
                logger.warning(f"Repetitive text sample: '{text[:100]}...'")
                logger.warning(
                    "Suggestion: If this issue persists, consider adjusting the audio input "
                    "or using a different transcription engine if available."
                )
        
        if word_timestamps:
            return segments_list, text.strip()
        else:
            return text.strip()

    def _transcribe_with_transformers(self, audio_data, rate, word_timestamps=False):
        if self.transformers_pipeline is None:
            logger.warning("Transformers pipeline not loaded. Returning empty transcription.")
            return "" if not word_timestamps else ([], "")
            
        # The pipeline expects a dictionary with raw data and sampling rate
        result = self.transformers_pipeline(
            {"raw": audio_data, "sampling_rate": rate},
            generate_kwargs={"language": self.language},
            return_timestamps="word" if word_timestamps else False,
            batch_size=1  # Force non-generator output for single items
        )
        
        # The pipeline result is a dictionary. Add explicit type check to satisfy Pylance.
        text_result = ""
        if isinstance(result, dict) and "text" in result:
            text_result = str(result.get("text", ""))

        if word_timestamps:
            # Reformat the output to be compatible with the orchestrator
            segments = []
            if isinstance(result, dict) and "chunks" in result and result["chunks"] is not None:
                for chunk in result["chunks"]:
                    # Create a mock segment object that has the necessary attributes
                    start, end = chunk["timestamp"]
                    word_obj = type('Word', (), {'word': chunk['text'], 'start': start, 'end': end})
                    segment_obj = type('Segment', (), {'text': chunk['text'], 'words': [word_obj], 'start': start, 'end': end})
                    segments.append(segment_obj)
            return segments, text_result.strip()
        else:
            return text_result.strip()
