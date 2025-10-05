# stellascript/audio/enhancement.py

import warnings
import numpy as np
import torch
import torchaudio
from df.enhance import enhance, init_df

class AudioEnhancer:
    def __init__(self, enhancement_method, device, rate):
        self.enhancement_method = enhancement_method
        self.device = device
        self.rate = rate
        self.demucs_model = None
        self.df_model = None
        self.df_state = None

    def apply(self, audio_data, is_live=False):
        """Apply selected audio enhancement method."""
        if self.enhancement_method == "none":
            return audio_data

        if self.enhancement_method == "deepfilternet":
            if self.df_model is None:
                print("Loading DeepFilterNet denoiser...")
                try:
                    self.df_model, self.df_state, _ = init_df()
                except Exception as e:
                    warnings.warn(f"Failed to load DeepFilterNet model: {e}. Skipping enhancement.")
                    return audio_data
            
            # Convert to torch tensor, add channel dimension for mono audio
            audio_tensor = torch.from_numpy(audio_data.copy()).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            if self.device == "cuda":
                audio_tensor = audio_tensor.to(self.device)

            # Resample to 48kHz for DeepFilterNet if necessary
            if self.rate != 48000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.rate, new_freq=48000
                ).to(self.device)
                audio_tensor = resampler(audio_tensor)

            # Enhance the audio
            if self.df_model is None or self.df_state is None:
                warnings.warn("DeepFilterNet model not loaded. Skipping enhancement.")
                return audio_data
            enhanced_audio = enhance(self.df_model, self.df_state, audio_tensor)

            # Resample back to the original rate if necessary
            if self.rate != 48000:
                resampler_back = torchaudio.transforms.Resample(
                    orig_freq=48000, new_freq=self.rate
                ).to(self.device)
                enhanced_audio = resampler_back(enhanced_audio)

            # Convert back to numpy array and remove channel dimension
            return enhanced_audio.squeeze(0).cpu().numpy()

        elif self.enhancement_method == "demucs":
            if is_live:
                 warnings.warn("Demucs is not recommended for live processing due to high latency. Using it anyway.")
            
            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
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
