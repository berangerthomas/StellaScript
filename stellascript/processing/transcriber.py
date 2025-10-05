# stellascript/processing/transcriber.py

import numpy as np
import torch
from faster_whisper import WhisperModel

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

        self._load_models()

    def _load_models(self):
        """Load transcription model(s) based on engine configuration."""
        if self.engine in ["faster-whisper", "auto"]:
            print("Loading faster-whisper model...")
            compute_type = "float16" if self.device.type == "cuda" else "float32"
            self.faster_whisper_model = WhisperModel(
                self.model_id, device=self.device.type, compute_type=compute_type
            )

        if self.engine in ["transformers", "auto"]:
            print("Loading transformers Whisper model...")
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            
            model_name = f"openai/whisper-{self.model_id}"
            self.transformers_processor = WhisperProcessor.from_pretrained(model_name)
            self.transformers_model = WhisperForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
            
            if self.device.type == "cuda":
                self.transformers_model = self.transformers_model.half()

        if self.engine == "auto":
            print(f"Auto-selection enabled: segments <{self.auto_engine_threshold}s will use transformers, >=<{self.auto_engine_threshold}s will use faster-whisper")
        elif self.engine == "faster-whisper":
            print("Using faster-whisper for all segments")
        else:
            print("Using transformers for all segments")

    def transcribe_segment(self, audio_data, rate, padding_duration):
        """Transcribe audio segment with intelligent engine selection."""
        audio_duration = len(audio_data) / rate
        
        if audio_duration < 0.5:
            print(f"DEBUG: Segment too short ({audio_duration:.2f}s), skipping")
            return ""
        
        padding_samples = int(padding_duration * rate)
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        
        padded_audio = np.concatenate([silence_padding, audio_data, silence_padding])
        padded_duration = len(padded_audio) / rate
        
        print(f"DEBUG: Original duration: {audio_duration:.2f}s, padded duration: {padded_duration:.2f}s")
        
        if self.engine == "auto":
            use_transformers = audio_duration < self.auto_engine_threshold
            engine_name = "transformers" if use_transformers else "faster-whisper"
            print(f"DEBUG: Auto-selecting {engine_name} for {audio_duration:.2f}s segment (threshold: {self.auto_engine_threshold}s)")
        elif self.engine == "transformers":
            use_transformers = True
        else:
            use_transformers = False
        
        if use_transformers:
            return self._transcribe_with_transformers(padded_audio, rate)
        else:
            return self._transcribe_with_faster_whisper(padded_audio)

    def _transcribe_with_faster_whisper(self, audio_data):
        """Transcribe with faster-whisper."""
        segments, _ = self.faster_whisper_model.transcribe(
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
        
        result_text = "".join(segment.text for segment in segments).strip()
        
        if result_text:
            words = result_text.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    print(f"WARNING: High repetition detected (unique ratio: {unique_ratio:.2f}) in faster-whisper output")
                    print(f"  -> Consider lowering --auto-engine-threshold (current: {self.auto_engine_threshold}s)")
        
        return result_text

    def _transcribe_with_transformers(self, audio_data, rate):
        """Transcribe with transformers."""
        audio_normalized = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        processed_input = self.transformers_processor(
            audio_normalized,
            sampling_rate=rate,
            return_tensors="pt"
        )
        features = processed_input.input_features.to(self.device)
        
        attention_mask = torch.ones(
            features.shape[:-1], dtype=torch.long, device=self.device
        )
        
        predicted_ids = self.transformers_model.generate(
            features,
            attention_mask=attention_mask,
            language=self.language,
            task="transcribe",
            max_length=448,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
        )
        
        transcription = self.transformers_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return transcription
