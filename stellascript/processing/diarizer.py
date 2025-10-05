# stellascript/processing/diarizer.py

import os
import torch
import numpy as np
import traceback
from pyannote.audio import Pipeline
from pyannote.core import Segment

class Diarizer:
    def __init__(self, device, method, hf_token, rate):
        self.device = device
        self.method = method
        self.rate = rate
        self.diarization_pipeline = self._load_diarization_pipeline(hf_token)
        self.vad_model = None
        self.vad_utils = None

    def _load_diarization_pipeline(self, hf_token):
        """Load the diarization pipeline."""
        if self.method != "pyannote":
            return None
        print("Loading diarization model...")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            if pipeline is None:
                raise RuntimeError("Pipeline.from_pretrained returned None.")

            if self.device.type == "cuda":
                pipeline.model = pipeline.model.half()
            return pipeline.to(self.device)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'eval'" in str(e):
                raise RuntimeError(
                    "Failed to load diarization pipeline. Check HUGGING_FACE_TOKEN and user agreement."
                ) from e
            else:
                raise e

    def _ensure_vad_loaded(self):
        """Lazy loading of VAD model."""
        if self.vad_model is None:
            print("Loading VAD model...")
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", 
                model="silero_vad", 
                force_reload=False
            )

    def diarize_pyannote(self, audio_data, min_speakers=None, max_speakers=None):
        """Diarize using pyannote pipeline."""
        print("Processing segment with 'pyannote' method...")
        diarization_params = {}
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
            
        diarization_result = self.diarization_pipeline(
            {"waveform": torch.from_numpy(audio_data).unsqueeze(0), "sample_rate": self.rate},
            **diarization_params
        )
        segments_list = list(diarization_result.itertracks(yield_label=True))
        unique_speakers = set(speaker_label for _, _, speaker_label in segments_list)
        print(f"Found {len(segments_list)} speech segments from {len(unique_speakers)} speakers.")
        return segments_list

    def diarize_cluster(self, audio_data, speaker_manager, similarity_threshold, max_speakers=None):
        """Diarize using VAD and clustering."""
        print("Segmenting speech with Silero VAD...")
        self._ensure_vad_loaded()
        (get_speech_timestamps, _, _, _, _) = self.vad_utils
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio_data), self.vad_model, sampling_rate=self.rate
        )
        print(f"VAD found {len(speech_timestamps)} speech segments.")

        print("Identifying speakers with clustering...")
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import normalize
        except ImportError:
            raise ImportError("scikit-learn is required. Please run: pip install scikit-learn")

        valid_segments_info = []
        audio_segments_for_batch = []
        for ts in speech_timestamps:
            start_samples, end_samples = ts["start"], ts["end"]
            audio_segment = audio_data[start_samples:end_samples]
            if len(audio_segment) < self.rate * 0.5: continue
            
            valid_segments_info.append({
                "turn": Segment(start_samples / self.rate, end_samples / self.rate),
                "audio_segment": audio_segment
            })
            audio_segments_for_batch.append(audio_segment)

        if not valid_segments_info:
            return [], 0

        all_embeddings = speaker_manager.get_embeddings(audio_segments_for_batch)

        if all_embeddings.size == 0:
            return [], 0

        if all_embeddings.ndim == 3 and all_embeddings.shape[1] == 1:
            all_embeddings = all_embeddings.squeeze(1)

        normalized_embeddings = normalize(all_embeddings, norm="l2", axis=1)
        clustering = AgglomerativeClustering(
            n_clusters=max_speakers if max_speakers else None,
            metric="cosine", linkage="average",
            distance_threshold=1 - similarity_threshold if not max_speakers else None
        ).fit(normalized_embeddings)
        
        cluster_labels = clustering.labels_
        found_speakers = len(set(cluster_labels))
        
        segments_with_speakers = [{
            "speaker_label": f"SPEAKER_{label:02d}",
            "turn": info["turn"],
            "audio_segment": info["audio_segment"]
        } for info, label in zip(valid_segments_info, cluster_labels)]
        
        return segments_with_speakers, found_speakers

    def apply_vad_to_chunk(self, audio_chunk):
        """Apply VAD to a small audio chunk (for live subtitle mode)."""
        self._ensure_vad_loaded()
        audio_tensor = torch.from_numpy(np.copy(audio_chunk))
        speech_prob = self.vad_model(audio_tensor, self.rate).item()
        return speech_prob
