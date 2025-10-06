# stellascript/processing/diarizer.py

import os
import torch
import numpy as np
import traceback
from typing import Any, Dict
from pyannote.audio import Pipeline
from pyannote.core import Segment
from ..logging_config import get_logger

logger = get_logger(__name__)

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
        logger.info("Loading pyannote diarization model")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if pipeline is None:
                raise RuntimeError("Pipeline.from_pretrained returned None.")

            if self.device.type == "cuda":
                pipeline.model = pipeline.model.half()
            logger.info("Pyannote diarization model loaded successfully")
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
            logger.info("Loading Silero VAD model")
            self.vad_model, self.vad_utils = torch.hub.load(  # type: ignore
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False
            )

    def diarize_pyannote(self, audio_data, min_speakers=None, max_speakers=None):
        """Diarize using pyannote pipeline."""
        if self.diarization_pipeline is None:
            raise RuntimeError(
                "Diarization pipeline not initialized. Check if method is 'pyannote'."
            )
        
        diarization_params = {}
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
        
        param_log = ", ".join([f"{k}={v}" for k, v in diarization_params.items()])
        logger.info(f"Processing audio with pyannote diarization ({param_log if param_log else 'default params'})")

        diarization_result = self.diarization_pipeline(
            {"waveform": torch.from_numpy(audio_data).unsqueeze(0), "sample_rate": self.rate},
            **diarization_params
        )
        annotation = diarization_result
        if not hasattr(annotation, "itertracks"):
            # The result is not an annotation object, so it's likely a container.
            # Let's find the annotation object within its attributes.
            for value in vars(annotation).values():
                if hasattr(value, "itertracks"):
                    annotation = value
                    break
            else:
                # This else block runs if the loop completes without finding an annotation.
                raise TypeError(f"Could not find an annotation object in the diarization result: {diarization_result}")
        segments_list = list(annotation.itertracks(yield_label=True))
        return segments_list

    def diarize_cluster(self, audio_data, speaker_manager, similarity_threshold, max_speakers=None):
        """Diarize using VAD and clustering."""
        logger.info("Segmenting speech with Silero VAD")
        self._ensure_vad_loaded()
        assert self.vad_utils is not None, "VAD utils should be loaded by _ensure_vad_loaded"
        get_speech_timestamps = self.vad_utils['get_speech_timestamps']
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio_data), self.vad_model, sampling_rate=self.rate
        )
        logger.info(f"VAD found {len(speech_timestamps)} potential speech segments")

        logger.info("Filtering and preparing segments for speaker embedding...")
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
            logger.warning("No valid speech segments found after filtering.")
            return [], 0
        
        logger.info(f"Extracted {len(audio_segments_for_batch)} valid segments for embedding.")
        all_embeddings = speaker_manager.get_embeddings(audio_segments_for_batch)

        if all_embeddings.size == 0:
            logger.warning("Speaker embedding extraction resulted in no embeddings.")
            return [], 0

        if all_embeddings.ndim == 3 and all_embeddings.shape[1] == 1:
            all_embeddings = all_embeddings.squeeze(1)

        logger.info("Identifying speakers with Agglomerative Clustering...")
        normalized_embeddings = normalize(all_embeddings, norm="l2", axis=1)
        
        clustering_params: Dict[str, Any] = {
            "metric": "cosine",
            "linkage": "average"
        }
        if max_speakers:
            clustering_params["n_clusters"] = max_speakers
            logger.info(f"Clustering with a fixed number of speakers: {max_speakers}")
        else:
            clustering_params["distance_threshold"] = 1 - similarity_threshold
            logger.info(f"Clustering with similarity threshold: {similarity_threshold}")

        clustering = AgglomerativeClustering(**clustering_params).fit(normalized_embeddings)
        
        cluster_labels = clustering.labels_
        found_speakers = len(set(cluster_labels))
        logger.info(f"Clustering identified {found_speakers} unique speakers.")
        
        segments_with_speakers = [{
            "speaker_label": f"SPEAKER_{label:02d}",
            "turn": info["turn"],
            "audio_segment": info["audio_segment"]
        } for info, label in zip(valid_segments_info, cluster_labels)]
        
        return segments_with_speakers, found_speakers

    def apply_vad_to_chunk(self, audio_chunk):
        """Apply VAD to a small audio chunk (for live subtitle mode)."""
        self._ensure_vad_loaded()
        assert self.vad_model is not None, "VAD model should be loaded by _ensure_vad_loaded"
        audio_tensor = torch.from_numpy(np.copy(audio_chunk))
        speech_prob = self.vad_model(audio_tensor, self.rate).item()
        return speech_prob
