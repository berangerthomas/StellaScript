# stellascript/orchestrator.py

import os
import queue
import threading
import time
import traceback
import warnings
import wave
from datetime import datetime, timedelta

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.core import Segment

from . import config
from .audio.capture import AudioCapture
from .audio.enhancement import AudioEnhancer
from .processing.diarizer import Diarizer
from .processing.speaker_manager import SpeakerManager
from .processing.transcriber import Transcriber

warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*huggingface_hub.*cache-system uses symlinks.*")
warnings.filterwarnings("ignore", message=".*Module 'speechbrain.pretrained' was deprecated.*")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_LOCAL_DIR_IS_SYMLINK_SUPPORTED"] = "0"

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
    ):
        print("Launching StellaScriptTranscription...")
        self.model_id = model_id
        self.language = language
        self.mode = mode
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.diarization_method = diarization_method
        self.enhancement_method = enhancement_method
        
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_TOKEN")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Instantiate modules
        self.transcriber = Transcriber(
            model_id=model_id,
            device=self.device,
            engine=transcription_engine,
            auto_engine_threshold=auto_engine_threshold,
            language=language,
        )
        self.diarizer = Diarizer(
            device=self.device,
            method=diarization_method,
            hf_token=hf_token,
            rate=config.RATE,
        )
        self.speaker_manager = SpeakerManager(
            device=self.device,
            similarity_threshold=similarity_threshold,
        )
        self.enhancer = AudioEnhancer(
            enhancement_method=enhancement_method,
            device=self.device,
            rate=config.RATE,
        )
        self.audio_capture = AudioCapture(
            format=config.FORMAT,
            channels=config.CHANNELS,
            rate=config.RATE,
            chunk=config.CHUNK,
        )
        
        print("All models loaded successfully.")
        self._setup_audio_config()
        self._setup_buffers_and_queues()

        self.chunk_timestamps = {}
        self.chunk_counter = 0
        self.is_running = False
        self.start_time = None
        self.filename = None
        self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

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
        self.audio_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=3)
        self.transcription_queue = queue.Queue()
        self._reset_buffers()

    def _reset_buffers(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.full_audio_buffer_list = []
        if self.mode == "subtitle":
            self.vad_speech_buffer = np.array([], dtype=np.float32)
            self.vad_is_speaking = False
            self.vad_silence_counter = 0

    def _init_output_file(self):
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
            print(f"Error during transcription of segment {chunk_id}:")
            traceback.print_exc()
        finally:
            if chunk_id in self.chunk_timestamps:
                del self.chunk_timestamps[chunk_id]

    def _process_transcription_cluster(self, chunk_id, audio_data):
        print(f"DEBUG: Processing segment {chunk_id} with live 'cluster' method.")
        try:
            embeddings = self.speaker_manager.get_embeddings([audio_data])
            if not embeddings: return
            embedding_np = embeddings[0]
        except Exception as e:
            print(f"DEBUG: Could not get embedding for segment {chunk_id}: {e}")
            return

        assigned_speaker = self.speaker_manager.get_speaker_id(embedding_np)
        if assigned_speaker is None:
            print(f"DEBUG: Could not assign speaker for segment {chunk_id}.")
            return

        print(f"DEBUG: Identified segment {chunk_id} as {assigned_speaker}")
        if len(audio_data) < int(0.5 * config.RATE): return

        transcription = self.transcriber.transcribe_segment(audio_data, config.RATE, config.TRANSCRIPTION_PADDING_S)
        if not transcription or transcription.isspace(): return

        chunk_start_time = self.chunk_timestamps.get(chunk_id, {}).get("start", 0)
        elapsed_seconds = chunk_start_time
        timestamp = self._calculate_video_timestamp(elapsed_seconds)
        line = f"[{timestamp}][{assigned_speaker}] {transcription}\n"
        print(f"Live: {line.strip()}")
        self._write_to_file(line, force_flush=True)

    def _process_transcription_pyannote(self, chunk_id, audio_data):
        segments_list = self.diarizer.diarize_pyannote(audio_data)
        merged_segments = self._merge_speaker_segments(segments_list, audio_data)
        for segment_info in merged_segments:
            self._transcribe_and_display(segment_info)

    def _merge_speaker_segments(self, segments_list, audio_data):
        merged_segments = []
        if not segments_list: return merged_segments

        valid_segments = []
        audio_segments_for_batch = []
        for turn, _, pyannote_speaker in segments_list:
            audio_segment = audio_data[int(turn.start * config.RATE):int(turn.end * config.RATE)]
            if len(audio_segment) < config.RATE * 0.5: continue
            valid_segments.append({"turn": turn, "audio_segment": audio_segment})
            audio_segments_for_batch.append(audio_segment)

        if not valid_segments: return merged_segments

        batch_embeddings_np = self.speaker_manager.get_embeddings(audio_segments_for_batch)
        
        identified_segments = []
        for idx, segment_info in enumerate(valid_segments):
            embedding_np = batch_embeddings_np[idx]
            assigned_speaker = self.speaker_manager.get_speaker_id(embedding_np)
            if assigned_speaker is None: continue
            identified_segments.append({**segment_info, "assigned_speaker": assigned_speaker})

        # Merging logic remains the same...
        merged_groups = []
        if identified_segments:
            current_group = [identified_segments[0]]
            for i in range(1, len(identified_segments)):
                current_segment = identified_segments[i]
                last_segment_in_group = current_group[-1]
                same_speaker = current_segment["assigned_speaker"] == last_segment_in_group["assigned_speaker"]
                time_gap = current_segment["turn"].start - last_segment_in_group["turn"].end
                should_merge = same_speaker and (self.mode == "transcription" or time_gap < config.MAX_MERGE_GAP_S)
                if should_merge:
                    current_group.append(current_segment)
                else:
                    merged_groups.append(current_group)
                    current_group = [current_segment]
            merged_groups.append(current_group)

        for group in merged_groups:
            if not group: continue
            speaker_id = group[0]["assigned_speaker"]
            start_time = group[0]["turn"].start
            end_time = group[-1]["turn"].end
            merged_audio = np.concatenate([s["audio_segment"] for s in group])
            merged_segments.append({
                "speaker_label": speaker_id,
                "turn": Segment(start_time, end_time),
                "audio_segment": merged_audio,
            })
        return merged_segments

    def _transcribe_and_display(self, segment_info):
        turn = segment_info["turn"]
        speaker_label = segment_info["speaker_label"]
        speaker_audio_segment = segment_info["audio_segment"]

        if len(speaker_audio_segment) < int(0.5 * config.RATE): return

        transcription = self.transcriber.transcribe_segment(speaker_audio_segment, config.RATE, config.TRANSCRIPTION_PADDING_S)
        if not transcription or transcription.isspace(): return

        chunk_start_time = self.chunk_timestamps.get(self.chunk_counter - 1, {}).get("start", 0)
        elapsed_seconds = chunk_start_time + turn.start
        timestamp = self._calculate_video_timestamp(elapsed_seconds)

        if self.mode == "transcription":
            if self.transcription_buffer["speaker"] == speaker_label:
                self.transcription_buffer["text"] += " " + transcription
            else:
                self._flush_transcription_buffer()
                self.transcription_buffer.update({"speaker": speaker_label, "timestamp": timestamp, "text": transcription})
        else:
            line = f"[{timestamp}][{speaker_label}] {transcription}\n"
            print(f"Live: {line.strip()}")
            self._write_to_file(line, force_flush=True)

    def _flush_transcription_buffer(self):
        if self.transcription_buffer["speaker"] and self.transcription_buffer["text"]:
            line = f"[{self.transcription_buffer['timestamp']}][{self.transcription_buffer['speaker']}] {self.transcription_buffer['text'].strip()}\n"
            print(f"Finalized: {line.strip()}")
            self._write_to_file(line, force_flush=True)
            self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

    def _create_optimal_transcription_chunks(self, segments_with_speakers):
        optimal_chunks = []
        current_chunk = {"speaker": None, "segments": [], "duration": 0.0}
        
        for i, segment_info in enumerate(segments_with_speakers):
            should_cut = False
            if current_chunk["speaker"] is None: pass
            elif current_chunk["speaker"] != segment_info["speaker_label"]: should_cut = True
            elif current_chunk["duration"] + (segment_info["turn"].end - segment_info["turn"].start) > config.MAX_CHUNK_DURATION_S: should_cut = True
            elif current_chunk["duration"] > config.TARGET_CHUNK_DURATION_S:
                if i < len(segments_with_speakers) - 1:
                    gap = segments_with_speakers[i + 1]["turn"].start - segment_info["turn"].end
                    if gap >= config.MIN_SILENCE_GAP_S: should_cut = True
        
            if should_cut and current_chunk["segments"]:
                chunk_start = current_chunk["segments"][0]["turn"].start
                chunk_end = current_chunk["segments"][-1]["turn"].end
                chunk_audio = np.concatenate([s["audio_segment"] for s in current_chunk["segments"]])
                optimal_chunks.append({"speaker_label": current_chunk["speaker"], "turn": Segment(chunk_start, chunk_end), "audio_segment": chunk_audio})
                current_chunk = {"speaker": None, "segments": [], "duration": 0.0}
            
            if current_chunk["speaker"] is None: current_chunk["speaker"] = segment_info["speaker_label"]
            current_chunk["segments"].append(segment_info)
            current_chunk["duration"] += segment_info["turn"].end - segment_info["turn"].start
        
        if current_chunk["segments"]:
            chunk_start = current_chunk["segments"][0]["turn"].start
            chunk_end = current_chunk["segments"][-1]["turn"].end
            chunk_audio = np.concatenate([s["audio_segment"] for s in current_chunk["segments"]])
            optimal_chunks.append({"speaker_label": current_chunk["speaker"], "turn": Segment(chunk_start, chunk_end), "audio_segment": chunk_audio})
        
        return optimal_chunks

    def _generate_filename(self, base_name=None, found_speakers=None):
        model_name_safe = self.model_id.replace("/", "_")
        base = os.path.splitext(os.path.basename(base_name))[0] if base_name else "live"
        parts = [base, self.mode, model_name_safe, self.diarization_method]
        if self.diarization_method == 'cluster':
            parts.append(f"thresh{self.speaker_manager.similarity_threshold:.2f}")
        if found_speakers is not None:
            parts.append(f"{found_speakers}-speakers")
        if not base_name:
            parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
        return "_".join(parts) + ".txt"

    def transcribe_file(self, file_path):
        print(f"Transcribing audio file: {file_path}")
        try:
            with wave.open(file_path, "rb") as wf:
                if wf.getframerate() != config.RATE: raise ValueError("Unsupported sample rate.")
                audio_bytes = wf.readframes(wf.getnframes())
            audio_float32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            audio_float32 = self.enhancer.apply(audio_float32, is_live=False)

            if self.diarization_method == "pyannote":
                raw_segments = self.diarizer.diarize_pyannote(audio_float32, self.min_speakers, self.max_speakers)
                segments_with_audio = [{"speaker_label": spk, "turn": turn, "audio_segment": audio_float32[int(turn.start*config.RATE):int(turn.end*config.RATE)]} for turn, _, spk in raw_segments]
                found_speakers = len(set(s["speaker_label"] for s in segments_with_audio))
            else: # cluster
                segments_with_audio, found_speakers = self.diarizer.diarize_cluster(audio_float32, self.speaker_manager, self.speaker_manager.similarity_threshold, self.max_speakers)

            self.filename = self._generate_filename(base_name=file_path, found_speakers=found_speakers)
            self._init_output_file()

            optimal_chunks = self._create_optimal_transcription_chunks(segments_with_audio)
            self.chunk_timestamps[0] = {"start": 0, "end": len(audio_float32) / config.RATE}
            for chunk_info in optimal_chunks:
                self._transcribe_and_display(chunk_info)

            self._flush_transcription_buffer()
            self._write_to_file("\n# Transcription complete.\n", force_flush=True)
            print(f"\nTranscription finished. Results saved to {self.filename}")

        except Exception as e:
            print(f"An error occurred during file transcription: {e}")
            traceback.print_exc()
        finally:
            if 0 in self.chunk_timestamps: del self.chunk_timestamps[0]

    def _transcribe_audio(self):
        while self.is_running or not self.result_queue.empty():
            try:
                chunk_id, audio_data = self.result_queue.get(timeout=0.5)
                self._process_transcription(chunk_id, audio_data)
            except queue.Empty:
                continue

    def _process_audio_stream(self, in_data, frame_count, time_info, status):
        now = datetime.now()
        chunk = np.frombuffer(in_data, dtype=np.float32)
        chunk = self.enhancer.apply(chunk, is_live=True)
        self.full_audio_buffer_list.append(chunk)
        if self.mode == "subtitle":
            self._process_subtitle_mode(chunk, now)
        else:
            self._process_transcription_mode(chunk, now)
        return (in_data, 0) # pyaudio.paContinue is 0

    def _process_subtitle_mode(self, chunk, now):
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
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        if len(self.audio_buffer) >= self.max_buffer_samples:
            buffer_duration = len(self.audio_buffer) / config.RATE
            ts_start = (now - timedelta(seconds=buffer_duration)).timestamp()
            self._add_audio_segment(self.chunk_counter, self.audio_buffer, ts_start, now.timestamp())
            self.audio_buffer = np.array([], dtype=np.float32)

    def start_recording(self):
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
        print("Starting audio stream...")
        if self.stream: self.stream.start_stream()

    def stop_recording(self):
        if not self.is_running: return
        print("\nStopping recording...")
        self.is_running = False
        if hasattr(self, "stream") and self.stream:
            self.audio_context.__exit__(None, None, None)
        if self.audio_buffer.size > 0:
            now = datetime.now()
            duration = len(self.audio_buffer) / config.RATE
            self._add_audio_segment(self.chunk_counter, self.audio_buffer, (now - timedelta(seconds=duration)).timestamp(), now.timestamp())
        if hasattr(self, "transcribe_thread"):
            self.transcribe_thread.join(timeout=120)
        if self.mode == "transcription": self._flush_transcription_buffer()
        self._write_to_file("", force_flush=True)
        print(f"Transcription saved to {self.filename}")

    def save_audio(self):
        if not self.full_audio_buffer_list: return
        full_audio = np.concatenate(self.full_audio_buffer_list)
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        scaled = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767.0)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(config.RATE)
            wf.writeframes(scaled.tobytes())
        print(f"Audio saved to {filename}")
