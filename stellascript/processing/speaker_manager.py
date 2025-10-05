# stellascript/processing/speaker_manager.py

import os
import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition

class SpeakerManager:
    def __init__(self, device, similarity_threshold):
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.embedding_model = self._load_speaker_embedding_model()
        self.speaker_embeddings_normalized = {}
        self.next_speaker_id = 1

    def _load_speaker_embedding_model(self):
        """Load speaker embedding model with Windows symlink workaround."""
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
                print("Windows symlink issue detected. Using copy strategy...")
                import speechbrain as sb
                original_strategy = getattr(sb.utils.fetching, "LOCAL_STRATEGY", None)
                sb.utils.fetching.LOCAL_STRATEGY = sb.utils.fetching.CopyStrategy()
                try:
                    embedding_model = SpeakerRecognition.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb"
                    )
                finally:
                    if original_strategy:
                        sb.utils.fetching.LOCAL_STRATEGY = original_strategy
            else:
                raise e
        
        if embedding_model:
            if self.device.type == "cuda":
                embedding_model = embedding_model.half()
            return embedding_model.to(self.device)
        raise RuntimeError("Failed to load speaker embedding model.")

    def get_speaker_id(self, embedding):
        """Get or assign speaker ID based on embedding similarity."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        embedding = np.array(embedding, dtype=np.float32, copy=True)
        embedding_flat = embedding.flatten()
        
        norm = np.linalg.norm(embedding_flat)
        if norm == 0:
            print("DEBUG: Warning - zero norm embedding, skipping")
            return None
        embedding_norm = embedding_flat / norm
        
        if not self.speaker_embeddings_normalized:
            speaker_id = f"SPEAKER_{self.next_speaker_id - 1:02d}"
            self.speaker_embeddings_normalized[speaker_id] = embedding_norm
            self.next_speaker_id += 1
            print(f"DEBUG: Created new speaker {speaker_id} (first speaker)")
            return speaker_id

        best_match = None
        best_similarity = -1.0

        for speaker_id, stored_embedding_norm in self.speaker_embeddings_normalized.items():
            similarity = float(np.dot(embedding_norm, stored_embedding_norm))
            print(f"DEBUG: Comparing with {speaker_id}: similarity = {similarity:.4f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        print(
            f"DEBUG: Best match: {best_match} with similarity {best_similarity:.4f}, threshold: {self.similarity_threshold}"
        )

        if best_similarity > self.similarity_threshold:
            existing_embedding = self.speaker_embeddings_normalized[best_match]
            weight = 0.7
            updated_embedding = (weight * existing_embedding) + ((1 - weight) * embedding_norm)
            updated_norm = np.linalg.norm(updated_embedding)
            if updated_norm > 0:
                self.speaker_embeddings_normalized[best_match] = updated_embedding / updated_norm
            
            print(f"DEBUG: Assigned to existing speaker {best_match} and updated their embedding.")
            return best_match
        else:
            speaker_id = f"SPEAKER_{self.next_speaker_id - 1:02d}"
            self.speaker_embeddings_normalized[speaker_id] = embedding_norm
            self.next_speaker_id += 1
            print(f"DEBUG: Created new speaker {speaker_id} (similarity too low)")
            return speaker_id

    def get_embeddings(self, audio_segments):
        """Get embeddings for a batch of audio segments."""
        if not audio_segments:
            return []

        try:
            max_len = max(len(seg) for seg in audio_segments)
            padded_segments = []
            for seg in audio_segments:
                if len(seg) < max_len:
                    padded = np.pad(seg, (0, max_len - len(seg)), mode='constant')
                else:
                    padded = seg
                padded_segments.append(padded)
            
            batch_tensor = torch.stack([
                torch.from_numpy(seg).float() for seg in padded_segments
            ]).to(self.device)
            
            batch_embeddings = self.embedding_model.encode_batch(batch_tensor)
            return batch_embeddings.cpu().numpy()
            
        except Exception as e:
            print(f"DEBUG: Batch embedding failed: {e}")
            # Fallback to sequential processing
            embeddings = []
            for seg in audio_segments:
                try:
                    tensor = torch.from_numpy(seg).float().unsqueeze(0).to(self.device)
                    embedding = self.embedding_model.encode_batch(tensor)
                    embeddings.append(embedding.cpu().numpy())
                except Exception as e2:
                    print(f"DEBUG: Failed to process a segment in fallback: {e2}")
                    embeddings.append(None) # Keep order
            return [emb for emb in embeddings if emb is not None]
