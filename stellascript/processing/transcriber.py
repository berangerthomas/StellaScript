# stellascript/processing/transcriber.py

"""
Handles audio transcription using the WhisperX library.

This module provides a Transcriber class that encapsulates the logic for loading
a Whisper model and using it to transcribe audio segments. It supports generating
both full-text transcriptions and detailed word-level timestamps.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import whisperx
from logging import getLogger

logger = getLogger(__name__)


class Transcriber:
    """
    A wrapper for the WhisperX transcription model.

    This class manages the loading of the WhisperX model and provides a simple
    interface to transcribe audio data. It can be configured for different
    model sizes, languages, and devices.
    """

    def __init__(self, model_id: str, device: torch.device, language: str) -> None:
        """
        Initializes the Transcriber.

        Args:
            model_id (str): The identifier of the Whisper model to use (e.g., 'large-v3').
            device (torch.device): The device to run the model on ('cuda' or 'cpu').
            language (str): The language of the audio to be transcribed.
        """
        self.device: torch.device = device
        self.language: str = language
        self.model_id: str = model_id
        self.whisperx_model: Optional[Any] = None
        self._load_models()

    def _load_models(self) -> None:
        """
        Loads the WhisperX transcription model.

        This method configures the model to use optimized compute types based on
        the available device (e.g., 'float16' for CUDA, 'int8' for CPU).

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            logger.info(f"Loading whisperx model '{self.model_id}'...")
            compute_type = "float16" if self.device.type == "cuda" else "int8"
            device_str = self.device.type
            self.whisperx_model = whisperx.load_model(
                self.model_id, device_str, compute_type=compute_type, language=self.language
            )
            logger.info(f"WhisperX model '{self.model_id}' loaded successfully on {device_str} with {compute_type} compute type.")
        except Exception as e:
            logger.exception(f"Failed to load whisperx model '{self.model_id}'.")
            raise RuntimeError(f"Failed to load whisperx model '{self.model_id}'.") from e

    def transcribe_segment(
        self,
        audio_data: np.ndarray,
        rate: int,
        padding_duration: float,
        word_timestamps: bool = False,
    ) -> Union[str, Tuple[List[Any], str]]:
        """
        Transcribes a single audio segment.

        The audio segment is padded with silence to improve transcription accuracy
        at the beginning and end of the speech.

        Args:
            audio_data (np.ndarray): The raw audio data of the segment.
            rate (int): The sample rate of the audio.
            padding_duration (float): The duration of silence padding in seconds.
            word_timestamps (bool): If True, returns word-level timestamps.

        Returns:
            Union[str, Tuple[List[Any], str]]:
                - If `word_timestamps` is False, returns the transcribed text as a string.
                - If `word_timestamps` is True, returns a tuple containing a list of
                  segment objects (with word details) and the full transcribed text.
        """
        audio_duration = len(audio_data) / rate
        
        if audio_duration < 0.5:
            logger.debug(f"Segment too short ({audio_duration:.2f}s), skipping")
            return "" if not word_timestamps else ([], "")
        
        padding_samples = int(padding_duration * rate)
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        
        padded_audio = np.concatenate([silence_padding, audio_data, silence_padding])
        
        return self._transcribe_with_whisperx(padded_audio, rate, word_timestamps)

    def _transcribe_with_whisperx(
        self,
        audio_data: np.ndarray,
        rate: int,
        word_timestamps: bool = False,
    ) -> Union[str, Tuple[List[Any], str]]:
        """
        Internal method to perform transcription using the WhisperX model.

        If word timestamps are requested, it also performs alignment to get
        precise timings for each word.

        Args:
            audio_data (np.ndarray): The audio data to transcribe.
            rate (int): The sample rate of the audio.
            word_timestamps (bool): Whether to generate word-level timestamps.

        Returns:
            Union[str, Tuple[List[Any], str]]: The transcription result.
        """
        if self.whisperx_model is None:
            logger.warning("WhisperX model not loaded. Returning empty transcription.")
            return "" if not word_timestamps else ([], "")

        result = self.whisperx_model.transcribe(audio_data, batch_size=16)
        
        if not word_timestamps:
            full_text = " ".join(seg.get("text", "") for seg in result.get("segments", [])).strip()
            return full_text

        try:
            device_str = self.device.type
            model_a, metadata = whisperx.load_align_model(language_code=self.language, device=device_str)
            result = whisperx.align(result["segments"], model_a, metadata, audio_data, device_str, return_char_alignments=False)
        except Exception as e:
            logger.error(f"Failed to align transcription: {e}. Returning non-aligned segments.")
            full_text = "".join(seg.get("text", "") for seg in result.get("segments", [])).strip()
            return [], full_text

        segments = []
        full_text = ""
        for segment in result.get("segments", []):
            words = []
            for word_info in segment.get("words", []):
                if not isinstance(word_info, dict): continue
                
                word, start, end = word_info.get("word"), word_info.get("start"), word_info.get("end")
                if word is None or start is None or end is None: continue

                word_obj = type('Word', (), {'word': word, 'start': float(start), 'end': float(end)})()
                words.append(word_obj)
            
            segment_text = segment.get("text", "").strip()
            segment_start, segment_end = segment.get("start"), segment.get("end")
            if segment_start is None or segment_end is None: continue
            
            segment_obj = type('Segment', (), {'text': segment_text, 'words': words, 'start': float(segment_start), 'end': float(segment_end)})()
            segments.append(segment_obj)
            full_text += segment_text + " "

        return segments, full_text.strip()
