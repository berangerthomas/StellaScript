# stellascript/processing/transcriber.py

import numpy as np
import torch
from logging import getLogger
import whisperx

logger = getLogger(__name__)

class Transcriber:
    def __init__(self, model_id, device, language):
        self.device = device
        self.language = language
        self.model_id = model_id
        self.whisperx_model = None
        self._load_models()

    def _load_models(self):
        """Load the WhisperX model."""
        try:
            logger.info(f"Loading whisperx model '{self.model_id}'...")
            # Implement CPU optimization: use 'int8' for CPU, 'float16' for CUDA
            compute_type = "float16" if self.device.type == "cuda" else "int8"
            # The underlying ctranslate2 library expects a string for the device, not a torch.device object.
            device_str = self.device.type
            self.whisperx_model = whisperx.load_model(
                self.model_id, device_str, compute_type=compute_type, language=self.language
            )
            logger.info(f"WhisperX model '{self.model_id}' loaded successfully on {device_str} with {compute_type} compute type.")
        except Exception:
            logger.exception(f"Failed to load whisperx model '{self.model_id}'.")
            raise

    def transcribe_segment(self, audio_data, rate, padding_duration, word_timestamps=False):
        """Transcribe audio segment using WhisperX."""
        audio_duration = len(audio_data) / rate
        
        if audio_duration < 0.5:
            logger.debug(f"Segment too short ({audio_duration:.2f}s), skipping")
            return "" if not word_timestamps else ([], "")
        
        padding_samples = int(padding_duration * rate)
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        
        padded_audio = np.concatenate([silence_padding, audio_data, silence_padding])
        
        return self._transcribe_with_whisperx(padded_audio, rate, word_timestamps)

    def _transcribe_with_whisperx(self, audio_data, rate, word_timestamps=False):
        """Transcribe with whisperx."""
        if self.whisperx_model is None:
            logger.warning("WhisperX model not loaded. Returning empty transcription.")
            return "" if not word_timestamps else ([], "")

        result = self.whisperx_model.transcribe(audio_data, batch_size=16)
        
        # If word timestamps are not needed, return the concatenated text and exit early.
        if not word_timestamps:
            full_text = " ".join(seg.get("text", "") for seg in result.get("segments", [])).strip()
            return full_text

        # Align the transcription for word-level timestamps
        try:
            # The underlying ctranslate2 library expects a string for the device, not a torch.device object.
            device_str = self.device.type
            model_a, metadata = whisperx.load_align_model(language_code=self.language, device=device_str)
            result = whisperx.align(result["segments"], model_a, metadata, audio_data, device_str, return_char_alignments=False)
        except Exception as e:
            logger.error(f"Failed to align transcription: {e}. Returning non-aligned segments.")
            # Fallback to non-aligned segments if alignment fails
            return [], "".join(seg.get("text", "") for seg in result.get("segments", [])).strip()

        # Reformat the output to be compatible with the orchestrator
        segments = []
        full_text = ""
        for segment in result["segments"]:
            words = []
            # Ensure 'words' key exists and is a list
            for word_info in segment.get("words", []):
                # Ensure word_info is a dictionary
                if not isinstance(word_info, dict): continue
                
                word = word_info.get("word", "")
                start = word_info.get("start") # Use .get for safety
                end = word_info.get("end")
                
                # Ensure timestamps are valid floats
                if start is None or end is None: continue

                # Create a simple object-like structure for words
                word_obj = type('Word', (), {'word': word, 'start': float(start), 'end': float(end)})()
                words.append(word_obj)
            
            segment_text = segment.get("text", "").strip()
            segment_start = segment.get("start")
            segment_end = segment.get("end")

            if segment_start is None or segment_end is None: continue
            
            segment_obj = type('Segment', (), {'text': segment_text, 'words': words, 'start': float(segment_start), 'end': float(segment_end)})()
            segments.append(segment_obj)
            full_text += segment_text + " "

        return segments, full_text.strip()
