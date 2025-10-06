# stellascript/orchestrator.py

import os
import queue
import threading
import time
import traceback
import warnings
import wave
from datetime import datetime, timedelta

from .logging_config import get_logger
from . import config

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_LOCAL_DIR_IS_SYMLINK_SUPPORTED"] = "0"

logger = get_logger(__name__)

class StellaScriptTranscription:
    def __init__(
        self,
        model_id="large-v3",
        language="fr",
        similarity_threshold=0.7,
        mode="transcription",
        min_speakers=None,
        max_speakers=None,
        diarization_method="pyannote",
        enhancement_method="none",
        transcription_engine="auto",
        auto_engine_threshold=15.0,
        save_enhanced_audio=False,
        save_recorded_audio=False,
    ):
        logger.info("StellaScriptTranscription orchestrator initialized (modules not loaded yet).")
        # Store configuration
        self.model_id = model_id
        self.language = language
        self.mode = mode
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.diarization_method = diarization_method
        self.similarity_threshold = similarity_threshold
        self.enhancement_method = enhancement_method
        self.transcription_engine = transcription_engine
        self.auto_engine_threshold = auto_engine_threshold
        self.save_enhanced_audio = save_enhanced_audio
        self.save_recorded_audio = save_recorded_audio

        # Initialize modules to None for lazy loading
        self.device = None
        self.transcriber = None
        self.diarizer = None
        self.speaker_manager = None
        self.enhancer = None
        self.audio_capture = None
        self.modules_initialized = False

        # Setup basic configurations
        self._setup_audio_config()
        self._setup_buffers_and_queues()
        self.chunk_timestamps = {}
        self.chunk_counter = 0
        self.is_running = False
        self.is_stopping = False
        self.start_time = None
        self.filename = None
        self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

    def _initialize_modules(self):
        """Initializes all heavy modules on demand."""
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
            engine=self.transcription_engine,
            auto_engine_threshold=self.auto_engine_threshold,
            language=self.language,
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

    def _setup_audio_config(self):
        if self.mode == "subtitle":
            self.max_buffer_duration = config.SUBTITLE_MAX_BUFFER_DURATION
            self.vad_speech_threshold = config.VAD_SPEECH_THRESHOLD
            self.vad_silence_duration_s = config.VAD_SILENCE_DURATION_S
            self.vad_min_speech_duration_s = config.VAD_MIN_SPEECH_DURATION_S
            self.vad_silence_samples = int(self.vad_silence_duration_s * config.RATE)
        else:
            self.max_buffer_duration = config.TRANSCRIPTION_MAX_BUFFER_DURATION
        self.max_buffer_samples = int(self.max_buffer_duration * config.RATE)

    def _setup_buffers_and_queues(self):
        """Initializes buffers, queues, and state variables."""
        self.result_queue = queue.Queue()
        self.transcription_queue = queue.Queue() # Add this line
        self.is_running = False
        self.is_stopping = False
        self.chunk_counter = 0
        self.chunk_timestamps = {}
        self.start_time = None
        self.buffer_start_time = None # For transcription mode
        self._reset_buffers()

    def _reset_buffers(self):
        """Resets audio buffers."""
        import numpy as np
        self.audio_buffer = np.array([], dtype=np.float32)
        self.full_audio_buffer_list = []
        if self.mode == "subtitle":
            self.vad_speech_buffer = np.array([], dtype=np.float32)
            self.vad_is_speaking = False
            self.vad_silence_counter = 0

    def _init_output_file(self):
        if self.filename:
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(f"# Transcription started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def get_transcription(self):
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return None

    def _add_audio_segment(self, chunk_id, audio_data, start_time, end_time):
        self.chunk_timestamps[chunk_id] = {"start": start_time, "end": end_time}
        self.result_queue.put((chunk_id, audio_data))
        self.chunk_counter += 1
        return self.chunk_counter - 1

    def _calculate_video_timestamp(self, elapsed_seconds):
        if elapsed_seconds < 0: elapsed_seconds = 0
        total_seconds = int(elapsed_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_to_file(self, line, force_flush=False):
        if self.filename:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(line)
                if force_flush:
                    f.flush()

    def _process_transcription(self, chunk_id, audio_data):
        try:
            is_live_mode = self.start_time is not None
            if is_live_mode and self.mode == "subtitle" and self.diarization_method == "cluster":
                self._process_transcription_cluster(chunk_id, audio_data)
            else:
                self._process_transcription_pyannote(chunk_id, audio_data)
        except Exception:
            logger.error(f"Error during transcription of segment {chunk_id}:")
            traceback.print_exc()
        finally:
            if chunk_id in self.chunk_timestamps:
                del self.chunk_timestamps[chunk_id]

    def _process_transcription_cluster(self, chunk_id, audio_data):
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

        transcription = self.transcriber.transcribe_segment(audio_data, config.RATE, config.TRANSCRIPTION_PADDING_S)
        if not transcription or transcription.isspace(): return

        chunk_start_time = self.chunk_timestamps.get(chunk_id, {}).get("start", 0)
        elapsed_seconds = chunk_start_time
        timestamp = self._calculate_video_timestamp(elapsed_seconds)
        line = f"[{timestamp}][{assigned_speaker}] {transcription}\n"
        logger.info(f"Live: {line.strip()}")
        self._write_to_file(line, force_flush=True)

    def _process_transcription_pyannote(self, chunk_id, audio_data):
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

    def _merge_consecutive_segments(self, segments_list):
        """
        Merges consecutive speech segments from the same speaker.
        """
        if not segments_list:
            return []
        
        import numpy as np
        from pyannote.core import Segment

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
            
            # In 'transcription' mode, always merge. In 'subtitle' mode, merge on small gaps.
            should_merge = is_same_speaker and (self.mode == "transcription" or time_gap < config.MAX_MERGE_GAP_S)

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
            logger.info("No consecutive segments from the same speaker to merge.")
        return final_segments

    def _transcribe_and_display(self, segment_info, total_segments=0, current_segment_num=0):
        assert self.transcriber is not None
        turn = segment_info["turn"]
        speaker_label = segment_info["speaker_label"]
        speaker_audio_segment = segment_info["audio_segment"]

        if len(speaker_audio_segment) < int(0.5 * config.RATE):
            return

        progress = f"({current_segment_num}/{total_segments})" if total_segments > 0 else ""
        logger.info(f"Transcribing segment {progress} for {speaker_label}...")

        transcription = self.transcriber.transcribe_segment(
            speaker_audio_segment, config.RATE, config.TRANSCRIPTION_PADDING_S
        )
        if not transcription or transcription.isspace():
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
            f"from {start_hms} to {end_hms}. Text: '{transcription[:80]}...'"
        )
        logger.debug(log_message)

        timestamp = self._calculate_video_timestamp(chunk_start_time + turn.start)

        if self.mode == "transcription":
            if self.transcription_buffer["speaker"] == speaker_label:
                self.transcription_buffer["text"] += " " + transcription
            else:
                self._flush_transcription_buffer()
                self.transcription_buffer.update({"speaker": speaker_label, "timestamp": timestamp, "text": transcription})
        else:
            line = f"[{timestamp}][{speaker_label}] {transcription}\n"
            logger.debug(f"Live: {line.strip()}")
            self._write_to_file(line, force_flush=True)

    def _flush_transcription_buffer(self):
        if self.transcription_buffer["speaker"] and self.transcription_buffer["text"]:
            line = f"[{self.transcription_buffer['timestamp']}][{self.transcription_buffer['speaker']}] {self.transcription_buffer['text'].strip()}\n"
            logger.debug(f"Finalized: {line.strip()}")
            self._write_to_file(line, force_flush=True)
            self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

    def _generate_filename(self, base_name=None, found_speakers=None):
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
            
        return "_".join(parts) + ".txt"

    def _save_enhanced_audio(self, original_path, audio_data, enhancement_method):
        """Saves the enhanced audio to a new file."""
        import numpy as np
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

    def _format_timedelta(self, td):
        """Formats a timedelta object into HH:MM:SS.ms."""
        total_seconds = td.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

    def transcribe_file(self, file_path):
        self._initialize_modules()
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
            if self.diarization_method == "pyannote":
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
            logger.info("************ 3/4 Merging Segments ************")
            merged_segments = self._merge_consecutive_segments(initial_segments)

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

    def _transcribe_audio(self):
        """Transcription thread target function."""
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

    def _process_audio_stream(self, in_data, frame_count, time_info, status):
        if self.is_stopping:
            return (in_data, 1) # pyaudio.paComplete is 1

        import numpy as np
        now = datetime.now()
        chunk = np.frombuffer(in_data, dtype=np.float32)
        assert self.enhancer is not None
        chunk = self.enhancer.apply(chunk, is_live=True)
        self.full_audio_buffer_list.append(chunk)
        if self.mode == "subtitle":
            self._process_subtitle_mode(chunk, now)
        else:
            self._process_transcription_mode(chunk, now)
        return (in_data, 0) # pyaudio.paContinue is 0

    def _process_subtitle_mode(self, chunk, now):
        import numpy as np
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

    def _process_transcription_mode(self, chunk, now):
        """Process audio in transcription mode with larger buffers."""
        import numpy as np
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

    def start_recording(self):
        """Initializes and starts the audio recording and processing threads."""
        self._initialize_modules()
        assert self.diarizer is not None
        assert self.audio_capture is not None
        if self.filename is None:
            self.filename = self._generate_filename()
            self._init_output_file()
        if self.mode == "subtitle": self.diarizer._ensure_vad_loaded()
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

    def stop_recording(self):
        if not self.is_running or self.is_stopping:
            return
        
        import numpy as np
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
        
        # Process the final VAD buffer for 'subtitle' mode
        if self.mode == "subtitle" and self.vad_speech_buffer.size > 0:
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
        
        # Flush the final merged text in 'transcription' mode
        if self.mode == "transcription":
            self._flush_transcription_buffer()
        
        self._write_to_file("\n# Transcription stopped.\n", force_flush=True)
        logger.info(f"Transcription finished. Results saved to {self.filename}")
        
        self.is_stopping = False

    def save_audio(self):
        if not self.full_audio_buffer_list: return
        import numpy as np
        full_audio = np.concatenate(self.full_audio_buffer_list)
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        scaled = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767.0)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(config.RATE)
            wf.writeframes(scaled.tobytes())
        logger.info(f"Audio saved to {filename}")
