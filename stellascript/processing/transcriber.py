# stellascript/processing/transcriber.py

import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
from logging import getLogger

logger = getLogger(__name__)

class Transcriber:
    def __init__(self, model_id, device, engine, auto_engine_threshold, language):
        self.device = device
        self.engine = engine
        self.auto_engine_threshold = auto_engine_threshold
        self.language = language
        self.model_id = model_id

        self.faster_whisper_model = None
        self.transformers_processor = None
        self.transformers_model = None
        self.transformers_pipeline = None # Add this line

        self._load_models()

    def _load_models(self):
        """Load transcription model(s) based on engine configuration."""
        if self.engine in ["faster-whisper", "auto"]:
            logger.info(f"Loading faster-whisper model '{self.model_id}'...")
            compute_type = "float16" if self.device.type == "cuda" else "float32"
            self.faster_whisper_model = WhisperModel(
                self.model_id, device=self.device.type, compute_type=compute_type
            )
            logger.info(f"Faster-whisper model '{self.model_id}' loaded successfully on {self.device.type} with {compute_type} compute type.")

        if self.engine in ["transformers", "auto"]:
            model_name = f"openai/whisper-{self.model_id}"
            logger.info(f"Loading transformers model '{model_name}'...")
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            self.transformers_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch_dtype,
                device=self.device,
            )
            logger.info(f"Transformers model '{model_name}' loaded successfully on {self.device.type} with {torch_dtype} dtype.")

        if self.engine == "auto":
            logger.info(f"Transcription engine mode: 'auto' (threshold: {self.auto_engine_threshold}s)")
            logger.info(f"Segments < {self.auto_engine_threshold}s will use 'transformers', longer segments will use 'faster-whisper'.")
        else:
            logger.info(f"Transcription engine mode: '{self.engine}' (all segments).")

    def transcribe_segment(self, audio_data, rate, padding_duration):
        """Transcribe audio segment with intelligent engine selection."""
        audio_duration = len(audio_data) / rate
        
        if audio_duration < 0.5:
            logger.debug(f"Segment too short ({audio_duration:.2f}s), skipping")
            return ""
        
        padding_samples = int(padding_duration * rate)
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        
        padded_audio = np.concatenate([silence_padding, audio_data, silence_padding])
        padded_duration = len(padded_audio) / rate
        
        if self.engine == "auto":
            use_transformers = audio_duration < self.auto_engine_threshold
            engine_name = "transformers" if use_transformers else "faster-whisper"
            logger.info(f"Transcribing a {audio_duration:.2f}s segment with '{engine_name}' (auto-selected).")
        elif self.engine == "transformers":
            use_transformers = True
            logger.info(f"Transcribing a {audio_duration:.2f}s segment with 'transformers'.")
        else:
            use_transformers = False
            logger.info(f"Transcribing a {audio_duration:.2f}s segment with 'faster-whisper'.")
        
        if use_transformers:
            return self._transcribe_with_transformers(padded_audio, rate)
        else:
            return self._transcribe_with_faster_whisper(padded_audio)

    def _transcribe_with_faster_whisper(self, audio_data):
        """Transcribe with faster-whisper."""
        if self.faster_whisper_model is None:
            logger.warning("faster-whisper model not loaded. Returning empty transcription.")
            return ""

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
            word_timestamps=False,
            prepend_punctuations="\"'¿([{-",
            append_punctuations="\"'.。,，!！?？:：)]}、",
        )
        text = "".join(segment.text for segment in segments)

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

        return text.strip()

    def _transcribe_with_transformers(self, audio_data, rate):
        if self.transformers_pipeline is None:
            logger.warning("Transformers pipeline not loaded. Returning empty transcription.")
            return ""
            
        # The pipeline expects a dictionary with raw data and sampling rate
        result = self.transformers_pipeline(
            {"raw": audio_data, "sampling_rate": rate},
            generate_kwargs={"language": self.language},
            batch_size=1  # Force non-generator output for single items
        )
        
        # The pipeline result is a dictionary. Add explicit type check to satisfy Pylance.
        text_result = ""
        if isinstance(result, dict) and "text" in result:
            text_result = str(result.get("text", ""))
        
        return text_result.strip()
