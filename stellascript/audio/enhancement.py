# stellascript/audio/enhancement.py

import os
import warnings
import numpy as np
import torch

class AudioEnhancer:
    def __init__(self, enhancement_method, device, rate):
        self.enhancement_method = enhancement_method
        self.device = device
        self.rate = rate
        self.nsnet2_session = None
        self.demucs_model = None

    def _ensure_nsnet2_model_downloaded(self):
        """Downloads the NSNet2 ONNX model if it doesn't exist."""
        model_path = "nsnet2-20ms-baseline.onnx"
        if not os.path.exists(model_path):
            print("Downloading NSNet2 model...")
            try:
                import urllib.request
                url = "https://github.com/microsoft/DNS-Challenge/raw/master/NSNet2-baseline/nsnet2-20ms-baseline.onnx"
                urllib.request.urlretrieve(url, model_path)
                print("NSNet2 model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download NSNet2 model: {e}")
        return model_path

    def apply(self, audio_data, is_live=False):
        """Apply selected audio enhancement method."""
        if self.enhancement_method == "none":
            return audio_data

        if self.enhancement_method == "nsnet2":
            if self.nsnet2_session is None:
                print("Loading NSNet2 denoiser...")
                try:
                    import onnxruntime as ort
                    model_path = self._ensure_nsnet2_model_downloaded()
                    self.nsnet2_session = ort.InferenceSession(model_path)
                except ImportError:
                    warnings.warn("onnxruntime is not installed. Please run 'uv sync'. Skipping enhancement.")
                    return audio_data
                except Exception as e:
                    warnings.warn(f"Failed to load NSNet2 model: {e}. Skipping enhancement.")
                    return audio_data
            
            audio = audio_data.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            frame_size = 320
            hop_size = 160
            
            output_audio = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                enhanced_frame = self.nsnet2_session.run(
                    None, 
                    {"input": frame.reshape(1, -1)}
                )[0]
                output_audio.append(enhanced_frame.flatten()[:hop_size])
            
            if not output_audio:
                return np.array([], dtype=np.float32)

            return np.concatenate(output_audio)

        elif self.enhancement_method == "demucs":
            if is_live:
                 warnings.warn("Demucs is not recommended for live processing due to high latency. Using it anyway.")
            
            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
                import torchaudio
            except ImportError as e:
                print(f"!!! DEMUCS IMPORT FAILED: {e} !!!")
                warnings.warn("Demucs not installed. Please run 'uv sync'. Skipping enhancement.")
                return audio_data

            if self.demucs_model is None:
                print("Loading Demucs model for audio separation...")
                self.demucs_model = get_model('htdemucs')
                self.demucs_model.to(self.device)
                self.demucs_model.eval()
            
            if self.rate != 44100:
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.rate,
                    new_freq=44100
                ).to(self.device)
                audio_tensor = resampler(audio_tensor.to(self.device))
            else:
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
            
            if audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.repeat(2, 1)
            
            audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
            
            print("Applying Demucs model...")
            with torch.no_grad():
                sources = apply_model(
                    self.demucs_model, 
                    audio_tensor,
                    split=True,
                    overlap=0.25
                )
            
            vocals = sources[0, 3]
            vocals_mono_tensor = vocals.mean(dim=0)
            
            if self.rate != 44100:
                resampler_back = torchaudio.transforms.Resample(
                    orig_freq=44100,
                    new_freq=self.rate
                ).to(self.device)
                vocals_mono_tensor = resampler_back(vocals_mono_tensor.unsqueeze(0)).squeeze()

            max_val = torch.max(torch.abs(vocals_mono_tensor))
            if max_val > 0:
                vocals_mono_tensor = vocals_mono_tensor / max_val

            vocals_mono = vocals_mono_tensor.cpu().numpy().astype(np.float32)
            
            print("Demucs processing complete.")
            return vocals_mono

        else:
            warnings.warn(f"Audio enhancement method '{self.enhancement_method}' is not yet implemented. Audio will not be processed.")
            return audio_data
