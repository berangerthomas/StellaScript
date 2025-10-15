<div align="center">
    <img src="docs/pictures/logo.jpg" alt="StellaScript Logo"/>
</div>
<div align="center">
    <a href="https://github.com/berangerthomas/stellascript/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
    <br/>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://huggingface.co/transformers"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow" alt="Transformers"></a>
</div>

<div align="center">
    <strong><a href="https://berangerthomas.github.io/StellaScript/">Read the documentation</a></strong>
</div>

-   Go directly to the [**Installation**](#installation) section.
-   See examples for [**live subtitling**](#scenario-1-live-subtitling-of-a-presentation) or [**transcribing a noisy field recording**](#scenario-3-transcribing-a-noisy-field-recording).

## A Python tool for transcription and speaker diarization

StellaScript is a Python application for generating speaker-aware transcriptions from live or pre-recorded audio. It integrates several machine learning models for its core functions: speech-to-text via OpenAI's Whisper model (using the `whisperx` library for optimized performance), speaker diarization using `pyannote.audio`, and speaker embedding generation with `SpeechBrain`.

## Core concepts and methodology

### Visual overview of the processing pipeline

The diagram below illustrates the complete processing pipeline, from audio input to the final formatted output. It shows how the different modules - such as audio enhancement, voice activity detection, diarization, and transcription - interact.

![StellaScript Processing Pipeline](docs/pictures/yed_pipeline_graph_vertical.png)

*Figure 1: A visual representation of the StellaScript processing pipeline, detailing the flow of data and the sequence of operations.*

### 1. Whisper Model Selection

The choice of the Whisper model is a trade-off between transcription accuracy (measured by Word Error Rate, or WER) and processing speed. The graph below shows the performance of different Whisper models across four languages, which can help in selecting the most appropriate model for a given task.

![Whisper Model Performance](docs/pictures/choosing_whisper_model_wer_vs_time.png)

*Figure 2: Word Error Rate (WER) versus processing time for various Whisper models in French, English, Spanish, and German.*

### Transcription Quality and Optimization

StellaScript is designed to produce transcriptions with a low error rate. The tool implements several techniques to reduce errors, such as voice activity detection and optimized decoding parameters. The technical details of these methods are available in the [**documentation**](https://berangerthomas.github.io/StellaScript/concepts/quality.html).

Here is a summary of the key optimization strategies:

-   **Voice Activity Detection (VAD)**: The system uses a VAD model (`Silero-VAD`) to identify speech segments. This process removes silences and non-speech noise, which can lead to transcription errors, including model hallucinations. The use of VAD is a common method for improving transcription quality by pre-processing the audio.

-   **Noise Reduction**: For noisy environments, StellaScript integrates neural network-based denoisers such as **`Demucs`**. This model separates speech from background noise to improve audio clarity before transcription.

-   **Decoding Parameters**: The tool is configured with selected decoding parameters (e.g., `beam_size=5`, `temperature=0.0`) intended to favor coherent and plausible output from the Whisper model.

-   **Hallucination Prevention**: In addition to silence padding, StellaScript includes mechanisms to mitigate hallucinations, such as setting thresholds on the model's log probability and compression ratio, which can indicate low-quality transcriptions.

Based on the performance data, here are some model selection recommendations for specific languages:

-   **French**: `large-v3` is recommended for the highest accuracy.
-   **English**: the `small` model offers a compelling trade-off, performing slightly better than `large-v3` while being nearly four times faster.
-   **Spanish**: the `medium` model provides a good balance between accuracy and processing speed.
-   **German**: `large-v3` is the best option for achieving the lowest error rate.

### 2. Hallucination mitigation via silence padding

To mitigate the risk of model hallucinations—where repetitive or nonsensical text is generated—the script employs a silence padding technique. This is one of several strategies used to ensure output quality. Before transcription, each audio chunk is wrapped with **1.5 seconds of digital silence** at both the beginning and end. Importantly, the final timestamps are adjusted to reflect the original audio, ensuring that this internal processing step *does not affect temporal accuracy*.

### 3. Speaker diarization methodologies

The tool offers two distinct methods for speaker identification, selectable via the `--diarization` argument:

-   **`pyannote` (Default)**: this method (`--diarization pyannote`) uses the pre-trained, end-to-end `pyannote/speaker-diarization-3.1` pipeline [[2]](#2). It is an end-to-end model that performs speech segmentation, embedding extraction, and clustering. It is often used for conversations with overlapping speech.

-   **`cluster`**: This method (`--diarization cluster`) is a multi-step pipeline:
    1.  **Voice Activity Detection (VAD)**: the audio is first segmented into speech and non-speech regions using the `Silero-VAD` model [[4]](#4).
    2.  **Speaker Embedding**: for each speech segment, a fixed-dimensional vector representation (an embedding, or "voiceprint") is extracted using the `speechbrain/spkrec-ecapa-voxceleb` model, which is based on the ECAPA-TDNN architecture [[3]](#3).
    3.  **Clustering**: all embeddings are grouped using agglomerative clustering based on cosine similarity to identify the unique speakers.

### 4. Intelligent chunking for file processing

When transcribing a file (using the `--file` argument), the pipeline must balance transcription quality with conversational context. The strategy differs based on the number of speakers:

#### Single-speaker optimization (`--max-speakers 1`)

1. The entire audio file is treated as a single segment, bypassing the diarization step.
2. This large segment is then chunked into fragments of 60-120 seconds.
3. Each chunk is transcribed independently, preserving context while maintaining Whisper's optimal input length.

This approach maximizes efficiency by eliminating unnecessary speaker identification when only one speaker is present.

#### Multi-speaker processing

1. The entire file is first diarized to identify all speaker turns, producing numerous small segments (typically 2-10 seconds each, corresponding to individual utterances).
2. These segments are then regrouped into larger chunks of 60-120 seconds while preserving:
   - Speaker boundaries (segments from different speakers are not merged)
   - Natural conversation breaks (silence gaps > 0.5s are respected)
   - Temporal coherence (consecutive segments from the same speaker are merged if separated by < 5s)
3. The resulting chunks are transcribed, providing Whisper with optimal context length.

#### Mode-specific behavior

- **`block` and `word` modes**: Apply the full chunking and merging pipeline described above to maximize transcription quality through contextual understanding.
- **`segment` mode**: Uses the original, smaller segments directly from the diarizer to minimize latency, which is essential for real-time subtitle generation.

This chunking methodology aims to provide each audio fragment sent for transcription with sufficient conversational context. This can contribute to the accuracy and readability of the output while respecting the constraints of each operational mode.

### 5. Output formatting modes

The final output format is controlled by the `--mode` argument, which dictates the granularity of the timestamps and the structure of the text.

-   **`block` Mode (Default)**: This mode is designed for readability. It performs a post-processing step that concatenates consecutive utterances from the same speaker. This transforms a fragmented stream of segments into coherent, paragraph-like blocks of text for each speaker, making the final document easy to read and analyze.

-   **`segment` Mode**: This mode is optimized for subtitles. It uses word-level timestamps to create shorter, time-accurate text segments that are ideal for captions. It balances readability with precise timing.

-   **`word` Mode**: This mode provides the highest level of detail. It generates a timestamp for every single word, which is useful for detailed analysis, research, or creating synchronized applications.

## Project Architecture

The application is built around a modular architecture with a central orchestrator managing the entire transcription pipeline.

-   **Entrypoint (`main.py`)**: The script that launches the application. It parses command-line arguments and initializes the main orchestrator.
-   **Orchestrator (`stellascript/orchestrator.py`)**: The core of the application. It coordinates all modules, from audio input to final text output, managing the different processing modes (`block`, `segment`, `word`).
-   **Configuration (`stellascript/config.py`)**: A centralized file holding all technical parameters, such as audio sample rates, buffer durations, and model settings, for easy tuning.

### Core Modules

-   **Audio Handling (`stellascript/audio/`)**:
    -   `capture.py`: Manages real-time audio recording from the microphone.
    -   `enhancement.py`: Applies noise reduction and audio cleaning models to improve source clarity.
-   **Processing (`stellascript/processing/`)**:
    -   `diarizer.py`: Identifies speaker segments in the audio stream ("who speaks when").
    -   `speaker_manager.py`: Creates and manages voiceprints to track unique speakers.
    -   `transcriber.py`: Converts audio segments into text using the Whisper model.

This structure separates concerns, making the system easier to maintain and extend.

## Installation

### Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv): A Python package installer.
-   A Hugging Face account and token.
-   **(Linux)**: `portaudio` development files may be required (`sudo apt-get install portaudio19-dev`).

### Setup

1.  **Clone repository**: `git clone https://github.com/berangerthomas/stellascript.git && cd stellascript`
2.  **Install dependencies**:
    ```bash
    uv venv
    uv sync
    ```
3.  **Activate environment**:
    -   Windows: `.venv\Scripts\activate`
    -   Linux/macOS: `source .venv/bin/activate`
4.  **Install PyTorch for GPU (optional)**:
    ```bash
    # Example for CUDA 12.6
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
5.  **Configure Hugging Face (optional)**: Create a `.env` file with your token:
    ```
    HUGGING_FACE_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    ```
    You must also accept the user agreements for `pyannote/speaker-diarization-3.1` on the Hugging Face Hub.

## Command-line reference

| Argument | Default | Description |
|---|---|---|
| `--language <lang>` | `en` | Language code (e.g., `en`, `es`, `de`, `fr`, `nl`, `pt`, `ro`, etc.). |
| `--model <id>` | `small` | Whisper model [[1]](#1). Choices: `tiny`, `base`, `small`, `medium`, `large`, `large-v1/v2/v3`, and distilled variants : `tiny.en`, `base.en`, `small.en`, `medium.en`, `distil-large-v2`, `distil-medium.en`, `distil-small.en`. |
| `--file <path>` | `None` | Path to a `.wav` file. If omitted, runs in live mode. |
| `--mode <mode>` | `block` | Output mode. Choices: `block`, `segment`, `word`. |
| `--diarization <method>` | `pyannote` | Diarization method. Choices: `pyannote`, `cluster`. |
| `--threshold <float>` | `0.7` | **(Cluster only)** Similarity threshold for speaker clustering (0.0-1.0). |
| `--min-speakers <int>` | `None` | **(File mode only)** Hint for the minimum number of speakers. |
| `--max-speakers <int>` | `None` | **(File mode only)** Hint for the maximum number of speakers. |
| `--enhancement <method>` | `none` | Audio enhancement. Choices: `none`, `deepfilternet` [[6]](#6), `demucs` [[5]](#5). |
| `--save-enhanced-audio` | `False` | Save the enhanced (denoised) audio to a new file. |
| `--save-recorded-audio` | `False` | Save the raw audio recorded from the microphone to a file. |

## Usage scenarios

### Scenario 1: live subtitling of a presentation

**Goal**: low-latency, real-time captions for a single speaker.
**Rationale**: `cluster` is faster than `pyannote`. A lower threshold (`0.4`) prevents voice modulation from creating a new speaker identity. The `small` model is selected for its strong balance of speed and accuracy in English. `segment` mode is used for time-accurate captions.
```bash
python main.py --mode segment --language en --diarization cluster --threshold 0.4 --model small
```

### Scenario 2: generating a transcript of a recorded interview

**Goal**: high-accuracy transcript of a 2-person conversation.
**Rationale**: `pyannote` is highly accurate for diarization. Specifying `--min-speakers 2 --max-speakers 2` constrains the model for optimal results. `block` mode creates a clean, readable document. `deepfilternet` provides lightweight noise reduction.
```bash
python main.py --mode block --file "interview.wav" --min-speakers 2 --max-speakers 2 --enhancement deepfilternet
```

### Scenario 3: transcribing a noisy field recording

**Goal**: extract intelligible speech from a noisy environment.
**Rationale**: `demucs` is a source separation model that can isolate vocals from background noise. This approach can be used to improve transcription clarity in challenging audio conditions.
```bash
python main.py --mode block --file "field_recording.wav" --enhancement demucs
```

### Scenario 4: processing a multi-speaker focus group recording

**Goal**: transcribe a complex conversation with an unknown number of speakers.
**Rationale**: `pyannote` excels in complex, multi-speaker scenarios. The default `block` mode will produce a clean, readable transcript.
```bash
python main.py --mode block --file "focus_group.wav" --diarization pyannote
```

## Output file naming

All transcriptions are saved to a `.txt` file with a name generated from the configuration settings.

-   **Format**: `{base_name}_{mode}_{model}_{diarization}_{details}_{timestamp}.txt`
-   **Components**:
    -   `base_name`: the original filename, or `live` for microphone recordings.
-   `mode`: `block`, `segment`, or `word`.
    -   `model`: the Whisper model used (e.g., `large-v3`).
    -   `diarization`: `pyannote` or `cluster`.
    -   `details`: `threshX.XX` for `cluster` mode, or `N-speakers` for file-based `pyannote` mode.
    -   `timestamp`: a `YYYYMMDD_HHMMSS` string added to live recordings.
-   **Examples**:
    -   `interview_block_large-v3_pyannote_2-speakers.txt`
    -   `live_segment_small_cluster_thresh0.60_20251002_213000.txt`

## Troubleshooting

### Repetitive or inaccurate text

-   **Symptom**: the transcription repeats a phrase or contains nonsensical text.
-   **Cause**: this can occur when short, ambiguous, or silent segments are passed to the transcription model. The silence padding implemented in this tool mitigates this, but it can still occasionally happen.
-   **Solution**:
    1.  Try a different Whisper model. Sometimes larger models are more prone to hallucination on certain types of audio, while smaller models might be less accurate.
    2.  Apply audio enhancement (`--enhancement deepfilternet` or `--enhancement demucs`) to clean up the audio before transcription.

### Incorrect speaker labels

-   **Symptom**: one person is labeled as multiple speakers, or multiple people are merged into one.
-   **Solution**:
    1.  If using `--diarization cluster`, adjust the `--threshold`. A lower value (e.g., `0.6`) makes it easier to merge speakers, while a higher value (e.g., `0.8`) makes it easier to separate them.
    2.  If using `--diarization pyannote` on a file, provide speaker hints with `--min-speakers` and `--max-speakers` to guide the model.

### Slow performance

-   **Symptom**: transcription speed is significantly slower than real-time.
-   **Solution**:
    1.  Ensure a GPU is available and that the GPU-accelerated version of PyTorch is installed correctly.
    2.  Use a smaller Whisper model (e.g., `--model medium` or `--model small`). The `small` model often provides an excellent balance of speed and accuracy.

## Bibliography

1.  <a name="1"></a>Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
2.  <a name="2"></a>Bredin, H., Yin, R., Coria, J. M., Gelly, G., Korshunov, P., Lavechin, M., Fustes, D., Titeux, H., Bouaziz, W., & Gill, M. (2020). *pyannote.audio: neural building blocks for speaker diarization*. [arXiv:1911.01255](https://arxiv.org/abs/1911.01255).
3.  <a name="3"></a>Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*. [arXiv:2005.07143](https://arxiv.org/abs/2005.07143).
4.  <a name="4"></a>Silero Team. (2024). *Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier*. [GitHub repository](https://github.com/snakers4/silero-vad).
5.  <a name="5"></a>Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2019). *Music Source Separation in the Waveform Domain*. [arXiv:1911.13254](https://arxiv.org/abs/1911.13254).
6.  <a name="6"></a>Schröter, H., Escalante-B., A. N., & Rosenkranz, T. (2022). *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*. [arXiv:2110.05588](https://arxiv.org/abs/2110.05588).

## Further Reading

For a deep dive into the methodologies for optimizing transcription quality, including detailed explanations of the techniques mentioned above, please refer to the **[Transcription Quality and Optimization](httpshttps://berangerthomas.github.io/StellaScript/concepts/quality.html)** page in our documentation.

## License

This project is licensed under the MIT License.
