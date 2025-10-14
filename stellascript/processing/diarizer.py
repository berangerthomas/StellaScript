# stellascript/processing/diarizer.py

"""
Handles speaker diarization using different methods like Pyannote and VAD with clustering.
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment

from ..logging_config import get_logger
from .speaker_manager import SpeakerManager

logger = get_logger(__name__)


class Diarizer:
    """
    A class to perform speaker diarization on audio data.

    This class supports multiple diarization methods, including the pre-trained
    Pyannote pipeline and a custom VAD-based clustering approach.
    """

    def __init__(self, device: torch.device, method: str, hf_token: Optional[str], rate: int) -> None:
        """
        Initializes the Diarizer.

        Args:
            device (torch.device): The device to run the models on.
            method (str): The diarization method to use ('pyannote', 'cluster').
            hf_token (Optional[str]): The Hugging Face authentication token for Pyannote.
            rate (int): The sample rate of the audio.
        """
        self.device: torch.device = device
        self.method: str = method
        self.rate: int = rate
        self.diarization_pipeline: Optional[Pipeline] = self._load_diarization_pipeline(hf_token)
        self.vad_model: Optional[Any] = None
        self.vad_utils: Optional[Dict[str, Any]] = None

    def _load_diarization_pipeline(self, hf_token: Optional[str]) -> Optional[Pipeline]:
        """
        Load the Pyannote diarization pipeline.

        Args:
            hf_token (Optional[str]): The Hugging Face token.

        Returns:
            Optional[Pipeline]: The loaded Pyannote pipeline, or None if not used.
        """
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

    def _ensure_vad_loaded(self) -> None:
        """Lazy loading of the Silero VAD model."""
        if self.vad_model is None:
            logger.info("Loading Silero VAD model")
            self.vad_model, self.vad_utils = torch.hub.load(  # type: ignore
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False
            )

    def diarize_pyannote(self, audio_data: np.ndarray, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> List[Tuple[Segment, str, str]]:
        """
        Diarize audio using the Pyannote pipeline.

        Args:
            audio_data (np.ndarray): The audio data to diarize.
            min_speakers (Optional[int]): The minimum number of speakers.
            max_speakers (Optional[int]): The maximum number of speakers.

        Returns:
            List[Tuple[Segment, str, str]]: A list of diarized segments.
        """
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

    def diarize_cluster(self, audio_data: np.ndarray, speaker_manager: SpeakerManager, similarity_threshold: float, max_speakers: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        Diarize audio using VAD and clustering.

        Args:
            audio_data (np.ndarray): The audio data to diarize.
            speaker_manager (SpeakerManager): The speaker manager for embeddings.
            similarity_threshold (float): The similarity threshold for clustering.
            max_speakers (Optional[int]): The maximum number of speakers.

        Returns:
            Tuple[List[Dict[str, Any]], int]: A tuple containing the list of
                                               diarized segments and the number
                                               of found speakers.
        """
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

    def apply_vad_to_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Apply VAD to a small audio chunk for live subtitle mode.

        Args:
            audio_chunk (np.ndarray): The audio chunk to process.

        Returns:
            float: The speech probability.
        """
        self._ensure_vad_loaded()
        assert self.vad_model is not None, "VAD model should be loaded by _ensure_vad_loaded"
        audio_tensor = torch.from_numpy(np.copy(audio_chunk))
        speech_prob = self.vad_model(audio_tensor, self.rate).item()
        return speech_prob
