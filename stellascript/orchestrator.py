# stellascript/orchestrator.py

"""
Main orchestrator for the Stellascript transcription pipeline.

This module contains the `StellaScriptTranscription` class, which coordinates
various components like audio capture, enhancement, diarization, and transcription
to provide a seamless real-time and file-based transcription service.
"""

import os
import queue
import threading
import time
import traceback
import warnings
import wave
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple


if TYPE_CHECKING:
    from .audio.capture import AudioCapture
    from .audio.enhancement import AudioEnhancer
    from .processing.diarizer import Diarizer
    from .processing.transcriber import Transcriber
    from .processing.speaker_manager import SpeakerManager

import numpy as np
import torch
from pyannote.core import Segment

from . import config
from .logging_config import get_logger

# Suppress Hugging Face Hub warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_LOCAL_DIR_IS_SYMLINK_SUPPORTED"] = "0"

logger = get_logger(__name__)


class StellaScriptTranscription:
    """
    Orchestrates the entire transcription process.

    This class manages the audio stream (from microphone or file),
    applies audio enhancement, performs speaker diarization, and uses a
    transcription model to convert speech to text. It handles different
    transcription modes (block, segment, word) and coordinates the various
    sub-modules.
    """

    def __init__(
        self,
        model_id: str = "large-v3",
        language: str = "fr",
        similarity_threshold: float = 0.7,
        mode: str = "block",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        diarization_method: str = "pyannote",
        enhancement_method: str = "none",
        save_enhanced_audio: bool = False,
        save_recorded_audio: bool = False,
    ) -> None:
        """
        Initializes the StellaScriptTranscription orchestrator.

        Args:
            model_id (str): The identifier of the Whisper model to use.
            language (str): The language for transcription.
            similarity_threshold (float): Threshold for speaker identification
                                          in 'cluster' mode.
            mode (str): The transcription mode ('block', 'segment', 'word').
            min_speakers (Optional[int]): Minimum number of speakers for Pyannote.
            max_speakers (Optional[int]): Maximum number of speakers.
            diarization_method (str): The diarization method ('pyannote', 'cluster').
            enhancement_method (str): The audio enhancement method.
            save_enhanced_audio (bool): Whether to save the enhanced audio.
            save_recorded_audio (bool): Whether to save the raw recorded audio.
        """
        logger.info("StellaScriptTranscription orchestrator initialized (modules not loaded yet).")
        # Store configuration
        self.model_id: str = model_id
        self.language: str = language
        self.mode: str = mode
        self.min_speakers: Optional[int] = min_speakers
        self.max_speakers: Optional[int] = max_speakers
        self.diarization_method: str = diarization_method
        self.similarity_threshold: float = similarity_threshold
        self.enhancement_method: str = enhancement_method
        self.save_enhanced_audio: bool = save_enhanced_audio
        self.save_recorded_audio: bool = save_recorded_audio

        # Initialize modules to None for lazy loading
        self.device: Optional[torch.device] = None
        self.transcriber: Optional['Transcriber'] = None
        self.diarizer: Optional['Diarizer'] = None
        self.speaker_manager: Optional['SpeakerManager'] = None
        self.enhancer: Optional['AudioEnhancer'] = None
        self.audio_capture: Optional['AudioCapture'] = None
        self.modules_initialized: bool = False

        # Setup basic configurations
        self._setup_audio_config()
        self._setup_buffers_and_queues()
        self.chunk_timestamps: Dict[int, Dict[str, float]] = {}
        self.chunk_counter: int = 0
        self.is_running: bool = False
        self.is_stopping: bool = False
        self.start_time: Optional[datetime] = None
        self.filename: Optional[str] = None
        self.transcription_buffer: Dict[str, Any] = {"speaker": None, "timestamp": None, "text": ""}

    def _initialize_modules(self) -> None:
        """
        Initializes all heavy modules on demand (lazy loading).

        This method imports and instantiates the main processing components like
        the transcriber, diarizer, and enhancer. This is done on the first
        call to a processing function to speed up initial application startup.
        """
        if self.modules_initialized:
            return

        logger.info("First use detected, initializing all modules...")

        # --- Defer heavy imports to this method ---
        import numpy as np
        import torch
        import torchaudio
        from dotenv import load_dotenv
        from .audio.capture import AudioCapture
        from .audio.enhancement import AudioEnhancer
        from .processing.diarizer import Diarizer
        from .processing.speaker_manager import SpeakerManager
        from .processing.transcriber import Transcriber
        
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_TOKEN")

        if self.diarization_method == "pyannote" and not hf_token:
            logger.warning("HUGGING_FACE_TOKEN not found. Falling back to 'cluster' diarization method.")
            self.diarization_method = "cluster"

            if self.min_speakers is not None:
                error_message = (
                    "HUGGING_FACE_TOKEN is missing, causing a fallback to 'cluster' diarization, "
                    "but '--min-speakers' is not supported in this mode. Please provide a token "
                    "or remove the '--min-speakers' argument."
                )
                logger.error(error_message)
                raise ValueError(error_message)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Instantiate modules
        self.transcriber = Transcriber(
            model_id=self.model_id,
            device=self.device,
            language=self.language
        )
        self.diarizer = Diarizer(
            device=self.device,
            method=self.diarization_method,
            hf_token=hf_token,
            rate=config.RATE,
        )
        self.speaker_manager = SpeakerManager(
            device=self.device,
            similarity_threshold=self.similarity_threshold,
        )
        self.enhancer = AudioEnhancer(
            enhancement_method=self.enhancement_method,
            device=self.device,
            rate=config.RATE,
        )
        self.audio_capture = AudioCapture(
            format=config.FORMAT,
            channels=config.CHANNELS,
            rate=config.RATE,
            chunk=config.CHUNK,
        )
        
        self.modules_initialized = True
        logger.info("All models loaded successfully.")

    def _setup_audio_config(self) -> None:
        """Sets up audio configuration based on the transcription mode."""
        if self.mode == "segment":
            self.max_buffer_duration: float = config.SUBTITLE_MAX_BUFFER_DURATION
            self.vad_speech_threshold: float = config.VAD_SPEECH_THRESHOLD
            self.vad_silence_duration_s: float = config.VAD_SILENCE_DURATION_S
            self.vad_min_speech_duration_s: float = config.VAD_MIN_SPEECH_DURATION_S
            self.vad_silence_samples: int = int(self.vad_silence_duration_s * config.RATE)
        else:
            self.max_buffer_duration = config.TRANSCRIPTION_MAX_BUFFER_DURATION
        self.max_buffer_samples: int = int(self.max_buffer_duration * config.RATE)

    def _setup_buffers_and_queues(self) -> None:
        """Initializes buffers, queues, and state variables."""
        self.result_queue: queue.Queue[Tuple[int, np.ndarray]] = queue.Queue()
        self.transcription_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False
        self.is_stopping = False
        self.chunk_counter = 0
        self.chunk_timestamps = {}
        self.start_time = None
        self.buffer_start_time: Optional[float] = None  # For transcription mode
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        """Resets audio buffers."""
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.full_audio_buffer_list: List[np.ndarray] = []
        if self.mode == "segment":
            self.vad_speech_buffer: np.ndarray = np.array([], dtype=np.float32)
            self.vad_is_speaking: bool = False
            self.vad_silence_counter: int = 0

    def _init_output_file(self) -> None:
        """Initializes the output file with a header."""
        if self.filename:
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(f"# Transcription started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def get_transcription(self) -> Optional[str]:
        """
        Retrieves a completed transcription line from the queue.

        Returns:
            Optional[str]: A transcribed line of text, or None if the queue is empty.
        """
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return None

    def _add_audio_segment(self, chunk_id: int, audio_data: np.ndarray, start_time: float, end_time: float) -> int:
        """
        Adds a new audio segment to the processing queue.

        Args:
            chunk_id (int): The unique identifier for the chunk.
            audio_data (np.ndarray): The audio data of the segment.
            start_time (float): The start timestamp of the segment.
            end_time (float): The end timestamp of the segment.

        Returns:
            int: The ID of the newly added chunk.
        """
        self.chunk_timestamps[chunk_id] = {"start": start_time, "end": end_time}
        self.result_queue.put((chunk_id, audio_data))
        self.chunk_counter += 1
        return self.chunk_counter - 1

    def _calculate_video_timestamp(self, elapsed_seconds: float) -> str:
        """
        Converts elapsed seconds into a HH:MM:SS timestamp format.

        Args:
            elapsed_seconds (float): The number of seconds elapsed.

        Returns:
            str: The formatted timestamp string.
        """
        if elapsed_seconds < 0:
            elapsed_seconds = 0
        total_seconds = int(elapsed_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_to_file(self, line: str, force_flush: bool = False) -> None:
        """
        Writes a line of text to the output file.

        Args:
            line (str): The text to write.
            force_flush (bool): If True, forces the file buffer to be written to disk.
        """
        if self.filename:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(line)
                if force_flush:
                    f.flush()

    def _process_transcription(self, chunk_id: int, audio_data: np.ndarray) -> None:
        """
        Routes an audio chunk to the appropriate transcription method.

        Args:
            chunk_id (int): The ID of the audio chunk.
            audio_data (np.ndarray): The audio data to process.
        """
        try:
            is_live_mode = self.start_time is not None
            if is_live_mode and self.mode == "segment" and self.diarization_method == "cluster":
                self._process_transcription_cluster(chunk_id, audio_data)
            else:
                self._process_transcription_pyannote(chunk_id, audio_data)
        except Exception:
            logger.error(f"Error during transcription of segment {chunk_id}:")
            traceback.print_exc()
        finally:
            if chunk_id in self.chunk_timestamps:
                del self.chunk_timestamps[chunk_id]

    def _process_transcription_cluster(self, chunk_id: int, audio_data: np.ndarray) -> None:
        """
        Processes transcription using the 'cluster' diarization method for live audio.

        Args:
            chunk_id (int): The ID of the audio chunk.
            audio_data (np.ndarray): The audio data to process.
        """
        assert self.speaker_manager is not None
        assert self.transcriber is not None
        logger.debug(f"Processing segment {chunk_id} with live 'cluster' method.")
        try:
            embeddings = self.speaker_manager.get_embeddings([audio_data])
            if not embeddings: return
            embedding_np = embeddings[0]
        except Exception as e:
            logger.debug(f"Could not get embedding for segment {chunk_id}: {e}")
            return

        assigned_speaker = self.speaker_manager.get_speaker_id(embedding_np)
        if assigned_speaker is None:
            logger.debug(f"Could not assign speaker for segment {chunk_id}.")
            return

        logger.debug(f"Identified segment {chunk_id} as {assigned_speaker}")
        if len(audio_data) < int(0.5 * config.RATE): return

        transcription_result = self.transcriber.transcribe_segment(audio_data, config.RATE, config.TRANSCRIPTION_PADDING_S)
        
        transcription = ""
        if isinstance(transcription_result, tuple):
            transcription = transcription_result[1]
        else:
            transcription = transcription_result

        if not transcription or transcription.isspace(): return

        chunk_start_time = self.chunk_timestamps.get(chunk_id, {}).get("start", 0)
        elapsed_seconds = chunk_start_time
        timestamp = self._calculate_video_timestamp(elapsed_seconds)
        line = f"[{timestamp}][{assigned_speaker}] {transcription}\n"
        logger.info(f"Live: {line.strip()}")
        self._write_to_file(line, force_flush=True)

    def _process_transcription_pyannote(self, chunk_id: int, audio_data: np.ndarray) -> None:
        """
        Processes transcription using the 'pyannote' diarization method.

        Args:
            chunk_id (int): The ID of the audio chunk.
            audio_data (np.ndarray): The audio data to process.
        """
        assert self.diarizer is not None
        pyannote_segments = self.diarizer.diarize_pyannote(audio_data)
        logger.debug(f"Diarization found {len(pyannote_segments)} segments.")
        
        initial_segments = []
        for turn, _, speaker in pyannote_segments:
            audio_segment = audio_data[int(turn.start * config.RATE):int(turn.end * config.RATE)]
            if len(audio_segment) < config.RATE * 0.5: continue
            initial_segments.append({
                "turn": turn, "speaker_label": speaker, "audio_segment": audio_segment
            })

        merged_segments = self._merge_consecutive_segments(initial_segments)
        for segment_info in merged_segments:
            self._transcribe_and_display(segment_info)

    def _merge_consecutive_segments(self, segments_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merges consecutive speech segments from the same speaker.

        Args:
            segments_list (List[Dict[str, Any]]): A list of segment dictionaries.

        Returns:
            List[Dict[str, Any]]: A new list of merged segment dictionaries.
        """
        if not segments_list:
            return []

        # Group consecutive segments from the same speaker.
        merged_groups = []
        # Ensure segments are sorted by start time
        segments_list.sort(key=lambda s: s["turn"].start)
        
        current_group = [segments_list[0]]
        for i in range(1, len(segments_list)):
            current_segment = segments_list[i]
            last_in_group = current_group[-1]
            
            is_same_speaker = current_segment["speaker_label"] == last_in_group["speaker_label"]
            time_gap = current_segment["turn"].start - last_in_group["turn"].end
            
            # Merge segments from the same speaker if the gap between them is small.
            should_merge = is_same_speaker and time_gap < config.MAX_MERGE_GAP_S

            if should_merge:
                current_group.append(current_segment)
            else:
                merged_groups.append(current_group)
                current_group = [current_segment]
        
        merged_groups.append(current_group)

        # Create the final merged segment objects.
        final_segments = []
        for group in merged_groups:
            if not group: continue
            
            speaker_id = group[0]["speaker_label"]
            start_time = group[0]["turn"].start
            end_time = group[-1]["turn"].end
            
            merged_audio = np.concatenate([s["audio_segment"] for s in group])
            
            final_segments.append({
                "speaker_label": speaker_id,
                "turn": Segment(start_time, end_time),
                "audio_segment": merged_audio,
            })
        
        if len(segments_list) != len(final_segments):
            logger.info(f"Merged {len(segments_list)} segments into {len(final_segments)} final segments.")
        else:
            logger.debug("No consecutive segments from the same speaker to merge.")
        return final_segments

    def _chunk_for_transcription(self, segments_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks diarized segments for transcription mode.

        It creates larger chunks of a target duration and then merges
        same-speaker segments within those chunks.

        Args:
            segments_list (List[Dict[str, Any]]): A list of segment dictionaries.

        Returns:
            List[Dict[str, Any]]: A new list of chunked and merged segments.
        """
        if not segments_list:
            return []

        logger.info(f"Chunking {len(segments_list)} segments for transcription...")
        all_merged_segments = []
        current_chunk_group = []
        current_chunk_duration = 0.0

        for i, segment in enumerate(segments_list):
            segment_duration = segment["turn"].end - segment["turn"].start

            # If adding the current segment would exceed the max duration, process the current chunk.
            if current_chunk_group and (current_chunk_duration + segment_duration > config.MAX_CHUNK_DURATION_S):
                logger.debug(f"Chunk full (max duration). Processing {len(current_chunk_group)} segments.")
                merged_in_chunk = self._merge_consecutive_segments(current_chunk_group)
                all_merged_segments.extend(merged_in_chunk)
                current_chunk_group = []
                current_chunk_duration = 0.0

            current_chunk_group.append(segment)
            current_chunk_duration += segment_duration

            # If the chunk is over the target duration, look for a good split point.
            if current_chunk_duration >= config.TARGET_CHUNK_DURATION_S:
                is_last_segment = (i == len(segments_list) - 1)
                if not is_last_segment:
                    next_segment = segments_list[i + 1]
                    is_speaker_change = segment["speaker_label"] != next_segment["speaker_label"]
                    silence_gap = next_segment["turn"].start - segment["turn"].end
                    is_long_silence = silence_gap > config.MIN_SILENCE_GAP_S

                    if is_speaker_change or is_long_silence:
                        logger.debug(f"Found split point. Processing {len(current_chunk_group)} segments.")
                        merged_in_chunk = self._merge_consecutive_segments(current_chunk_group)
                        all_merged_segments.extend(merged_in_chunk)
                        current_chunk_group = []
                        current_chunk_duration = 0.0
        
        # Process the final remaining chunk
        if current_chunk_group:
            logger.debug(f"Processing final chunk of {len(current_chunk_group)} segments.")
            merged_in_chunk = self._merge_consecutive_segments(current_chunk_group)
            all_merged_segments.extend(merged_in_chunk)

        logger.info(f"Re-chunked into {len(all_merged_segments)} segments for transcription.")
        return all_merged_segments

    def _transcribe_and_display(self, segment_info: Dict[str, Any], total_segments: int = 0, current_segment_num: int = 0) -> None:
        """
        Transcribes a single audio segment and handles its display or storage.

        Args:
            segment_info (Dict[str, Any]): Information about the segment to transcribe.
            total_segments (int): The total number of segments for progress tracking.
            current_segment_num (int): The number of the current segment.
        """
        assert self.transcriber is not None
        turn: Segment = segment_info["turn"]
        speaker_label = segment_info["speaker_label"]
        speaker_audio_segment = segment_info["audio_segment"]

        if len(speaker_audio_segment) < int(0.5 * config.RATE):
            return

        progress = f"({current_segment_num}/{total_segments})" if total_segments > 0 else ""
        logger.info(f"Transcribing segment {progress} for {speaker_label}...")

        # Word timestamps are needed for 'word' mode and for 'segment' mode (to re-segment subtitles).
        word_timestamps_enabled = self.mode in ["word", "segment"]

        transcription_result = self.transcriber.transcribe_segment(
            speaker_audio_segment,
            config.RATE,
            config.TRANSCRIPTION_PADDING_S,
            word_timestamps=word_timestamps_enabled
        )

        segments, full_text = [], ""
        if isinstance(transcription_result, tuple):
            segments, full_text = transcription_result
        else:
            full_text = transcription_result

        if not isinstance(full_text, str) or not full_text.strip():
            logger.warning(f"Transcription for segment {progress} resulted in empty text.")
            return

        chunk_start_time = self.chunk_timestamps.get(self.chunk_counter - 1, {}).get("start", 0)
        
        start_time_obj = timedelta(seconds=chunk_start_time + turn.start)
        end_time_obj = timedelta(seconds=chunk_start_time + turn.end)
        
        start_hms = self._format_timedelta(start_time_obj)
        end_hms = self._format_timedelta(end_time_obj)
        
        duration = turn.end - turn.start
        
        log_message = (
            f"Transcribed a {duration:.2f}s segment with [{self.transcriber.model_id}], "
            f"from {start_hms} to {end_hms}. Text: '{full_text[:80]}...'"
        )
        logger.debug(log_message)

        if self.mode == "word":
            self._write_word_level_timestamps(segments, speaker_label, chunk_start_time + turn.start)
        elif self.mode == "segment":
            self._segment_and_write_subtitles(segments, speaker_label, chunk_start_time + turn.start)
        elif self.mode == "block":
            timestamp = self._calculate_video_timestamp(chunk_start_time + turn.start)
            if self.transcription_buffer["speaker"] == speaker_label:
                self.transcription_buffer["text"] += " " + full_text
            else:
                self._flush_transcription_buffer()
                self.transcription_buffer.update({"speaker": speaker_label, "timestamp": timestamp, "text": full_text})
        else: # Fallback for safety, should not be reached
            timestamp = self._calculate_video_timestamp(chunk_start_time + turn.start)
            line = f"[{timestamp}][{speaker_label}] {full_text}"
            print(line)
            self.transcription_queue.put(line)
            logger.debug(f"Live: {line.strip()}")
            self._write_to_file(line + "\n", force_flush=True)

    def _segment_and_write_subtitles(self, segments: List[Any], speaker_label: str, chunk_start_time: float) -> None:
        """
        Re-segments a transcription based on word timestamps and writes subtitle lines.

        Args:
            segments (List[Any]): A list of transcription segments with word timestamps.
            speaker_label (str): The label of the speaker.
            chunk_start_time (float): The start time of the parent audio chunk.
        """
        all_words: List[Any] = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                all_words.extend(segment.words)

        if not all_words:
            logger.warning("Word timestamps not available for re-segmentation. Falling back to full segment.")
            if segments:
                full_text = " ".join(s.text for s in segments).strip()
                if full_text:
                    timestamp = self._calculate_video_timestamp(chunk_start_time + segments[0].start)
                    line_to_write = f"[{timestamp}][{speaker_label}] {full_text}\n"
                    self._write_to_file(line_to_write, force_flush=True)
            return

        logger.info(f"Re-segmenting transcription for {speaker_label} based on word timestamps...")
        line_count = 0
        current_line = ""
        line_start_time = all_words[0].start

        for i, word in enumerate(all_words):
            current_line += word.word + " "
            is_last_word = (i == len(all_words) - 1)
            next_word_start = all_words[i + 1].start if not is_last_word else float('inf')
            silence_duration = next_word_start - word.end
            line_duration = word.end - line_start_time
            
            if (silence_duration > config.SUBTITLE_MAX_SILENCE_S or
                (len(current_line) > config.SUBTITLE_MAX_LENGTH and word.word.endswith((' ', '.', ',', '?'))) or
                line_duration > config.SUBTITLE_MAX_DURATION_S or
                is_last_word):

                timestamp = self._calculate_video_timestamp(chunk_start_time + line_start_time)
                line_to_write = f"[{timestamp}][{speaker_label}] {current_line.strip()}\n"
                self._write_to_file(line_to_write, force_flush=True)
                line_count += 1
                
                if not is_last_word:
                    current_line = ""
                    line_start_time = next_word_start
        
        logger.info(f"Generated {line_count} subtitle lines from the original segment.")

    def _write_word_level_timestamps(self, segments: List[Any], speaker_label: str, chunk_start_time: float) -> None:
        """
        Writes word-level timestamps to the output file.

        Args:
            segments (List[Any]): A list of transcription segments with word timestamps.
            speaker_label (str): The label of the speaker.
            chunk_start_time (float): The start time of the parent audio chunk.
        """
        self._write_to_file(f"[{speaker_label}]\n", force_flush=True)
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    start_time = self._format_timedelta(timedelta(seconds=chunk_start_time + word.start))
                    end_time = self._format_timedelta(timedelta(seconds=chunk_start_time + word.end))
                    line = f"[{start_time} -> {end_time}] {word.word.strip()}\n"
                    self._write_to_file(line, force_flush=True)

    def _flush_transcription_buffer(self) -> None:
        """Writes the content of the transcription buffer to the file."""
        if self.transcription_buffer["speaker"] and self.transcription_buffer["text"]:
            line = f"[{self.transcription_buffer['timestamp']}][{self.transcription_buffer['speaker']}] {self.transcription_buffer['text'].strip()}\n"
            logger.debug(f"Finalized: {line.strip()}")
            self._write_to_file(line, force_flush=True)
            self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

    def _generate_filename(self, base_name: Optional[str] = None, found_speakers: Optional[int] = None) -> str:
        """
        Generates a descriptive filename for the transcription output.

        Args:
            base_name (Optional[str]): The base name from the input file.
            found_speakers (Optional[int]): The number of speakers detected.

        Returns:
            str: The generated filename.
        """
        model_name_safe = self.model_id.replace("/", "_")
        base = os.path.splitext(os.path.basename(base_name))[0] if base_name else "live"
        
        parts = [base, self.mode, model_name_safe, self.diarization_method]

        details = []
        if self.diarization_method == 'pyannote':
            if self.min_speakers is not None:
                details.append(f"min{self.min_speakers}")
            if self.max_speakers is not None:
                details.append(f"max{self.max_speakers}")
        elif self.diarization_method == 'cluster':
            details.append(f"thresh{self.similarity_threshold:.2f}")
            if self.max_speakers is not None:
                details.append(f"max{self.max_speakers}")

        if found_speakers is not None:
            details.append(f"{found_speakers}-speakers")

        if details:
            parts.append("_".join(details))

        if not base_name:
            parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))

        # Filter out any None or empty strings before joining
        filename = "_".join(filter(None, parts)) + ".txt"
        logger.debug(f"Generated filename: {filename}")
        return filename

    def _save_enhanced_audio(self, original_path: str, audio_data: np.ndarray, enhancement_method: str) -> None:
        """
        Saves the enhanced audio to a new file.

        Args:
            original_path (str): The path of the original audio file.
            audio_data (np.ndarray): The enhanced audio data.
            enhancement_method (str): The name of the enhancement method used.
        """
        try:
            base, ext = os.path.splitext(original_path)
            new_path = f"{base}_cleaned_{enhancement_method}{ext}"
            
            scaled = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            
            with wave.open(new_path, 'wb') as wf:
                wf.setnchannels(config.CHANNELS)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(config.RATE)
                wf.writeframes(scaled.tobytes())
            
            logger.info(f"Enhanced audio saved to {new_path}")
        except Exception as e:
            logger.error(f"Error saving enhanced audio: {e}")

    def _format_timedelta(self, td: timedelta) -> str:
        """
        Formats a timedelta object into HH:MM:SS.ms.

        Args:
            td (timedelta): The timedelta object to format.

        Returns:
            str: The formatted time string.
        """
        total_seconds = td.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

    def transcribe_file(self, file_path: str) -> None:
        """
        Transcribes an entire audio file.

        This method loads an audio file, applies enhancement and diarization,
        and then transcribes the resulting audio segments.

        Args:
            file_path (str): The path to the audio file to transcribe.
        """
        try:
            self._initialize_modules()
        except RuntimeError as e:
            logger.error(f"Failed to initialize modules: {e}")
            return
        assert self.enhancer is not None
        assert self.diarizer is not None
        assert self.speaker_manager is not None
        logger.info(f"Starting transcription for file: {file_path}")
        
        import torchaudio
        import torch
        import numpy as np

        try:
            # --- Audio Loading and Enhancement ---
            logger.info("************ 1/4 Audio Loading and Enhancement ************")
            audio_data, rate = torchaudio.load(file_path)
            duration_seconds = audio_data.shape[-1] / rate
            logger.info(f"Loaded audio file '{os.path.basename(file_path)}' ({duration_seconds:.2f}s)")

            if rate != config.RATE:
                logger.info(f"Resampling audio from {rate}Hz to {config.RATE}Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=config.RATE)
                audio_data = resampler(audio_data)
            
            if audio_data.dim() > 1 and audio_data.shape[0] > 1:
                logger.info("Audio has multiple channels, converting to mono.")
                audio_data = torch.mean(audio_data, dim=0, keepdim=True)

            audio_data_np = audio_data.squeeze().numpy().astype(np.float32)
            logger.info(f"Applying audio enhancement: '{self.enhancement_method}'")
            enhanced_audio = self.enhancer.apply(audio_data_np)
            if self.save_enhanced_audio and self.enhancement_method != 'none':
                self._save_enhanced_audio(file_path, enhanced_audio, self.enhancement_method)

            # --- Diarization ---
            logger.info(f"************ 2/4 Diarization by [{self.diarization_method}] ************")
            initial_segments = []
            found_speakers = 0
            
            # Intelligent Diarization: Skip if only one speaker is expected.
            if self.max_speakers == 1:
                logger.info("max_speakers is set to 1, skipping diarization.")
                from pyannote.core import Segment
                # Treat the entire audio as a single segment from one speaker.
                full_duration = len(enhanced_audio) / config.RATE
                initial_segments = [{
                    "turn": Segment(0, full_duration),
                    "speaker_label": "SPEAKER_00",
                    "audio_segment": enhanced_audio
                }]
                found_speakers = 1
            elif self.diarization_method == "pyannote":
                pyannote_segments = self.diarizer.diarize_pyannote(
                    enhanced_audio, self.min_speakers, self.max_speakers
                )
                if pyannote_segments:
                    found_speakers = len(set(s[2] for s in pyannote_segments))
                    logger.info(f"Pyannote found {len(pyannote_segments)} speech segments from {found_speakers} speakers.")
                else:
                    logger.warning("Pyannote did not find any speech segments.")

                # Standardize the output
                for turn, _, speaker in pyannote_segments:
                    audio_segment = enhanced_audio[int(turn.start * config.RATE):int(turn.end * config.RATE)]
                    if len(audio_segment) < config.RATE * 0.5: continue
                    initial_segments.append({
                        "turn": turn, "speaker_label": speaker, "audio_segment": audio_segment
                    })
            elif self.diarization_method == "cluster":
                logger.info("Starting clustering-based diarization...")
                initial_segments, found_speakers = self.diarizer.diarize_cluster(
                    enhanced_audio, self.speaker_manager, self.similarity_threshold, self.max_speakers
                )
                logger.info(f"Clustering found {len(initial_segments)} speech segments from {found_speakers} speakers.")

            self.filename = self._generate_filename(file_path, found_speakers)
            self._init_output_file()

            # --- Merge Consecutive Segments ---
            logger.info("************ 3/4 Merging/Chunking Segments ************")
            if self.mode == 'segment':
                # For segment (subtitle) mode, use segments as they are from the diarizer for lower latency.
                merged_segments = initial_segments
                logger.info("Segment mode: Using diarizer's original segmentation.")
            else:
                # For block and word modes, perform chunking for higher context.
                merged_segments = self._chunk_for_transcription(initial_segments)

            # --- Transcription and Buffering ---
            logger.info(f"************ 4/4 Transcription by [{self.model_id}] ************")
            total_segments = len(merged_segments)
            for i, segment_info in enumerate(merged_segments):
                self._transcribe_and_display(segment_info, total_segments, i + 1)

            # --- Finalize ---
            self._flush_transcription_buffer()
            self._write_to_file("\n# Transcription complete.\n", force_flush=True)
            logger.info(f"\nTranscription finished. Results saved to {self.filename}")

        except Exception as e:
            logger.error(f"An error occurred during file transcription: {e}")
            traceback.print_exc()
        finally:
            if 0 in self.chunk_timestamps: del self.chunk_timestamps[0]

    def _transcribe_audio(self) -> None:
        """
        Target function for the transcription thread.

        This method runs in a separate thread, continuously pulling audio
        segments from a queue and processing them.
        """
        while self.is_running or not self.result_queue.empty():
            try:
                # Use a timeout to prevent blocking indefinitely if the queue is empty
                # but is_running is still true for a short period.
                chunk_id, audio_data = self.result_queue.get(timeout=0.5)
                self._process_transcription(chunk_id, audio_data)
                self.result_queue.task_done()
            except queue.Empty:
                # If the queue is empty and we are no longer running, we can exit.
                if not self.is_running:
                    break
                continue
        
        logger.info("Transcription thread has finished processing all segments.")

    def _process_audio_stream(self, in_data: bytes, frame_count: int, time_info: Dict[str, float], status: int) -> Tuple[bytes, int]:
        """
        PyAudio stream callback function.

        This method is called by PyAudio for each new chunk of audio data
        from the microphone.

        Args:
            in_data (bytes): The audio data buffer.
            frame_count (int): The number of frames in the buffer.
            time_info (Dict[str, float]): Dictionary containing timestamps.
            status (int): PortAudio status flags.

        Returns:
            Tuple[bytes, int]: A tuple containing the audio data and a flag
                               indicating whether to continue the stream.
        """
        if self.is_stopping:
            return (in_data, 1)  # pyaudio.paComplete is 1

        now = datetime.now()
        chunk = np.frombuffer(in_data, dtype=np.float32)
        assert self.enhancer is not None
        chunk = self.enhancer.apply(chunk, is_live=True)
        self.full_audio_buffer_list.append(chunk)
        if self.mode == "segment":
            self._process_segment_mode(chunk, now)
        else:
            self._process_long_buffer_mode(chunk, now)
        return (in_data, 0) # pyaudio.paContinue is 0

    def _process_segment_mode(self, chunk: np.ndarray, now: datetime) -> None:
        """
        Processes an audio chunk in 'segment' (subtitle) mode using VAD.

        Args:
            chunk (np.ndarray): The incoming audio chunk.
            now (datetime): The current timestamp.
        """
        assert self.diarizer is not None
        speech_prob = self.diarizer.apply_vad_to_chunk(chunk)
        if speech_prob > self.vad_speech_threshold:
            self.vad_silence_counter = 0
            if not self.vad_is_speaking: self.vad_is_speaking = True
            self.vad_speech_buffer = np.concatenate([self.vad_speech_buffer, chunk])
        else:
            if self.vad_is_speaking:
                self.vad_silence_counter += len(chunk)
                if self.vad_silence_counter >= self.vad_silence_samples:
                    if len(self.vad_speech_buffer) / config.RATE > self.vad_min_speech_duration_s:
                        buffer_duration = len(self.vad_speech_buffer) / config.RATE
                        ts_start = (now - timedelta(seconds=buffer_duration)).timestamp()
                        self._add_audio_segment(self.chunk_counter, self.vad_speech_buffer, ts_start, now.timestamp())
                    self.vad_speech_buffer = np.array([], dtype=np.float32)
                    self.vad_is_speaking = False
                    self.vad_silence_counter = 0
        
        # Force processing if buffer is too long, even without silence
        if self.vad_is_speaking and (len(self.vad_speech_buffer) > self.max_buffer_samples):
            logger.debug(f"Segment mode buffer is full ({len(self.vad_speech_buffer) / config.RATE:.2f}s). Processing chunk.")
            buffer_duration = len(self.vad_speech_buffer) / config.RATE
            ts_start = (now - timedelta(seconds=buffer_duration)).timestamp()
            self._add_audio_segment(self.chunk_counter, self.vad_speech_buffer, ts_start, now.timestamp())
            self.vad_speech_buffer = np.array([], dtype=np.float32)
            # We don't reset vad_is_speaking here, as speech continues.

    def _process_long_buffer_mode(self, chunk: np.ndarray, now: datetime) -> None:
        """
        Processes an audio chunk in 'block' or 'word' mode with a long buffer.

        Args:
            chunk (np.ndarray): The incoming audio chunk.
            now (datetime): The current timestamp.
        """
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        
        # Store the timestamp when the buffer starts filling
        if self.buffer_start_time is None:
            self.buffer_start_time = now.timestamp()

        buffer_duration = len(self.audio_buffer) / config.RATE
        
        if buffer_duration >= config.TRANSCRIPTION_MAX_BUFFER_DURATION:
            ts_start = self.buffer_start_time
            ts_end = now.timestamp()
            
            self._add_audio_segment(
                self.chunk_counter, 
                self.audio_buffer.copy(),
                ts_start,
                ts_end
            )
            
            # Reset buffer and its start time for the next segment
            self.audio_buffer = np.array([], dtype=np.float32)
            self.buffer_start_time = None

    def start_recording(self) -> None:
        """
        Initializes and starts the audio recording and processing threads.
        """
        try:
            self._initialize_modules()
        except RuntimeError as e:
            logger.error(f"Failed to initialize modules: {e}")
            return
        assert self.diarizer is not None
        assert self.audio_capture is not None
        if self.filename is None:
            self.filename = self._generate_filename()
            self._init_output_file()
        if self.mode == "segment": self.diarizer._ensure_vad_loaded()
        self.is_running = True
        self.start_time = datetime.now()
        self._reset_buffers()
        self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()
        
        self.audio_context = self.audio_capture.audio_stream(callback=self._process_audio_stream)
        self.stream = self.audio_context.__enter__()
        logger.info("Starting audio stream...")
        if self.stream: self.stream.start_stream()

    def stop_recording(self) -> None:
        """
        Stops the audio recording and waits for all processing to complete.
        """
        if not self.is_running or self.is_stopping:
            return

        logger.info("\nStopping recording...")
        self.is_stopping = True # Signal to stop processing new audio chunks
        
        # Stop the audio stream first to prevent new data from the microphone
        if hasattr(self, "stream") and self.stream and self.stream.is_active():
            try:
                # This will stop the callback from being called
                self.audio_context.__exit__(None, None, None)
                logger.info("Audio stream stopped.")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        
        # Now that the stream is stopped, signal the transcription thread it can exit once the queue is empty
        self.is_running = False
        
        # Process the final audio buffer for 'transcription' mode
        if self.audio_buffer.size > 0:
            now = datetime.now()
            duration = len(self.audio_buffer) / config.RATE
            ts_start = (now - timedelta(seconds=duration)).timestamp()
            self._add_audio_segment(
                self.chunk_counter, 
                self.audio_buffer.copy(),
                ts_start, 
                now.timestamp()
            )
            logger.info(f"Queued final audio buffer ({duration:.2f}s) for processing.")
            self.audio_buffer = np.array([], dtype=np.float32)
        
        # Process the final VAD buffer for 'segment' mode
        if self.mode == "segment" and self.vad_speech_buffer.size > 0:
            if len(self.vad_speech_buffer) / config.RATE > self.vad_min_speech_duration_s:
                now = datetime.now()
                duration = len(self.vad_speech_buffer) / config.RATE
                ts_start = (now - timedelta(seconds=duration)).timestamp()
                self._add_audio_segment(
                    self.chunk_counter,
                    self.vad_speech_buffer.copy(),
                    ts_start,
                    now.timestamp()
                )
                logger.info(f"Queued final VAD speech buffer ({duration:.2f}s) for processing.")
                self.vad_speech_buffer = np.array([], dtype=np.float32)
        
        # Wait for the transcription thread to process all items in the queue
        if hasattr(self, "transcribe_thread"):
            logger.info("Waiting for transcription thread to finish...")
            self.transcribe_thread.join(timeout=10) # Wait up to 10 seconds
            if self.transcribe_thread.is_alive():
                logger.warning("Transcription thread timed out. Some segments may be lost.")
        
        # Flush the final merged text in 'block' mode
        if self.mode == "block":
            self._flush_transcription_buffer()
        
        self._write_to_file("\n# Transcription stopped.\n", force_flush=True)
        logger.info(f"Transcription finished. Results saved to {self.filename}")
        
        self.is_stopping = False

    def save_audio(self) -> None:
        """
        Saves the entire recorded audio session to a WAV file.
        """
        if not self.full_audio_buffer_list:
            return
        full_audio = np.concatenate(self.full_audio_buffer_list)
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        scaled = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767.0)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(config.RATE)
            wf.writeframes(scaled.tobytes())
        logger.info(f"Audio saved to {filename}")
