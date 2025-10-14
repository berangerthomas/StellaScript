# stellascript/processing/speaker_manager.py

# stellascript/processing/speaker_manager.py

"""
Manages speaker identification and embedding storage.

This module is responsible for loading a speaker recognition model, generating
embeddings for audio segments, and assigning speaker IDs based on similarity.
It maintains a registry of known speakers and their corresponding embeddings.
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition

from ..logging_config import get_logger

logger = get_logger(__name__)


class SpeakerManager:
    """
    Handles speaker embeddings and identification.

    This class uses a pre-trained speaker recognition model to create vector
    embeddings from audio segments. It can then compare these embeddings to
    identify known speakers or register new ones.
    """

    def __init__(self, device: torch.device, similarity_threshold: float) -> None:
        """
        Initializes the SpeakerManager.

        Args:
            device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
            similarity_threshold (float): The cosine similarity threshold for
                                          identifying a speaker.
        """
        self.device: torch.device = device
        self.similarity_threshold: float = similarity_threshold
        self.embedding_model: SpeakerRecognition = self._load_speaker_embedding_model()
        self.speaker_embeddings_normalized: Dict[str, np.ndarray] = {}
        self.next_speaker_id: int = 1

    def _load_speaker_embedding_model(self) -> SpeakerRecognition:
        """
        Loads the speaker embedding model.

        This method includes a workaround for a known symlink issue on Windows
        by temporarily changing the local fetching strategy in SpeechBrain.

        Returns:
            SpeakerRecognition: The loaded speaker recognition model.

        Raises:
            RuntimeError: If the model fails to load for any reason.
        """
        os.environ["SPEECHBRAIN_CACHE_DIR"] = os.path.join(
            os.getcwd(), "speechbrain_cache"
        )
        embedding_model = None
        try:
            embedding_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
        except OSError as e:
            if "privilège nécessaire" in str(e) or "WinError 1314" in str(e):
                logger.info("Windows symlink issue detected - using copy strategy")
                import speechbrain.utils.fetching
                original_strategy = getattr(speechbrain.utils.fetching, "LOCAL_STRATEGY", None)
                copy_strategy_class = getattr(speechbrain.utils.fetching, "CopyStrategy")
                setattr(speechbrain.utils.fetching, "LOCAL_STRATEGY", copy_strategy_class())
                try:
                    embedding_model = SpeakerRecognition.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb"
                    )
                finally:
                    if original_strategy:
                        setattr(speechbrain.utils.fetching, "LOCAL_STRATEGY", original_strategy)
            else:
                raise e
        
        if embedding_model:
            if self.device.type == "cuda":
                embedding_model = embedding_model.half()
            return embedding_model.to(self.device)
        raise RuntimeError("Failed to load speaker embedding model.")

    def get_speaker_id(self, embedding: Union[np.ndarray, torch.Tensor]) -> Optional[str]:
        """
        Gets or assigns a speaker ID based on embedding similarity.

        Compares the provided embedding with stored embeddings of known speakers.
        If a match is found above the similarity threshold, the existing speaker ID
        is returned. Otherwise, a new speaker is registered.

        Args:
            embedding (Union[np.ndarray, torch.Tensor]): The speaker embedding to identify.

        Returns:
            Optional[str]: The assigned speaker ID, or None if the embedding is invalid.
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        embedding = np.array(embedding, dtype=np.float32, copy=True)
        embedding_flat = embedding.flatten()
        
        norm = np.linalg.norm(embedding_flat)
        if norm == 0:
            logger.debug("Zero norm embedding detected - skipping")
            return None
        embedding_norm = embedding_flat / norm

        if not self.speaker_embeddings_normalized:
            speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
            self.speaker_embeddings_normalized[speaker_id] = embedding_norm
            self.next_speaker_id += 1
            logger.info(f"Registered first speaker as {speaker_id}.")
            return speaker_id

        best_match: Optional[str] = None
        best_similarity: float = -1.0

        for speaker_id, stored_embedding_norm in self.speaker_embeddings_normalized.items():
            similarity = float(np.dot(embedding_norm, stored_embedding_norm))
            logger.debug(f"Comparing with {speaker_id}: similarity = {similarity:.4f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        logger.debug(f"Best match: {best_match} with similarity {best_similarity:.4f} (threshold: {self.similarity_threshold})")

        if best_similarity > self.similarity_threshold and best_match is not None:
            logger.info(f"Segment assigned to existing speaker {best_match} with similarity {best_similarity:.4f}.")
            
            existing_embedding = self.speaker_embeddings_normalized[best_match]
            weight = 0.7
            updated_embedding = (weight * existing_embedding) + ((1 - weight) * embedding_norm)
            
            updated_norm = np.linalg.norm(updated_embedding)
            if updated_norm > 0:
                self.speaker_embeddings_normalized[best_match] = updated_embedding / updated_norm
                logger.debug(f"Updated embedding for {best_match}.")

            return best_match
        else:
            speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
            self.speaker_embeddings_normalized[speaker_id] = embedding_norm
            self.next_speaker_id += 1
            logger.info(f"Similarity {best_similarity:.4f} is below threshold. Registered new speaker: {speaker_id}")
            return speaker_id

    def get_embeddings(self, audio_segments: List[np.ndarray]) -> np.ndarray:
        """
        Gets embeddings for a batch of audio segments.

        This method attempts to process all segments in a single batch for efficiency.
        If batch processing fails, it falls back to processing segments one by one.

        Args:
            audio_segments (List[np.ndarray]): A list of audio segments as NumPy arrays.

        Returns:
            np.ndarray: A NumPy array of embeddings for the processed segments.
        """
        if not audio_segments:
            return np.array([])

        try:
            max_len = max(len(seg) for seg in audio_segments)
            padded_segments = [
                np.pad(seg, (0, max_len - len(seg)), mode='constant') if len(seg) < max_len else seg
                for seg in audio_segments
            ]
            
            batch_tensor = torch.stack([
                torch.from_numpy(seg).float() for seg in padded_segments
            ]).to(self.device)
            
            batch_embeddings = self.embedding_model.encode_batch(batch_tensor)
            return batch_embeddings.cpu().numpy()
            
        except Exception as e:
            logger.debug(f"Batch embedding failed: {e}. Falling back to sequential processing.")
            embeddings = []
            for seg in audio_segments:
                try:
                    tensor = torch.from_numpy(seg).float().unsqueeze(0).to(self.device)
                    embedding = self.embedding_model.encode_batch(tensor)
                    embeddings.append(embedding.cpu().numpy())
                except Exception as e2:
                    logger.debug(f"Failed to process segment in fallback: {e2}")
            
            if not embeddings:
                return np.array([])
            return np.concatenate(embeddings, axis=0)
