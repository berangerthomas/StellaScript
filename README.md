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

## A Python tool for transcription and speaker diarization

StellaScript is a Python application for generating speaker-aware transcriptions from live or pre-recorded audio. It integrates several machine learning models for its core functions: speech-to-text via OpenAI's Whisper model (using `faster-whisper` and Hugging Face `transformers` implementations), speaker diarization using `pyannote.audio`, and speaker embedding generation with `SpeechBrain`.

## Core concepts and methodology

### Visual overview of the processing pipeline

The diagram below illustrates the complete processing pipeline, from audio input to the final formatted output. It shows how the different modules - such as audio enhancement, voice activity detection, diarization, and transcription - interact.

![StellaScript Processing Pipeline](docs/pictures/yed_pipeline_graph_horizontal.png)

*Figure 1: A visual representation of the StellaScript processing pipeline, detailing the flow of data and the sequence of operations.*

### 1. Hybrid transcription strategy

The choice of transcription engine is critical for balancing speed and accuracy.

-   **`faster-whisper`**: a reimplementation of Whisper using CTranslate2, offering significant performance gains (3-5x faster on GPU) through quantization and optimized computation. It excels on long, clear audio segments.
-   **`transformers`**: the official Hugging Face implementation. While slower, it exhibits greater stability on short, ambiguous, or noisy audio segments, reducing the risk of repetitive hallucinations.

The default **`auto`** engine (`--transcription-engine auto`) leverages the strengths of both. It processes audio segments shorter than a configurable duration (`--auto-engine-threshold`) with `transformers` and longer segments with `faster-whisper`. This hybrid approach provides a robust balance suitable for most use cases.

The choice of the Whisper model is a trade-off between transcription accuracy (measured by Word Error Rate, or WER) and processing speed. The graph below shows the performance of different Whisper models across four languages, which can help in selecting the most appropriate model for a given task.

![Whisper Model Performance](docs/pictures/choosing_whisper_model_wer_vs_time.png)

*Figure 2: Word Error Rate (WER) versus processing time for various Whisper models in French, English, Spanish, and German.*

Based on the performance data, here are some model selection recommendations for specific languages:

-   **French**: `large-v3` is recommended for the highest accuracy.
-   **English**: the `small` model offers a compelling trade-off, performing slightly better than `large-v3` while being nearly four times faster.
-   **Spanish**: the `medium` model provides a good balance between accuracy and processing speed.
-   **German**: `large-v3` is the best option for achieving the lowest error rate.

### 2. Hallucination mitigation via silence padding

To mitigate the risk of model hallucinations—where repetitive or nonsensical text is generated—the script employs a silence padding technique. Before transcription, each audio chunk is wrapped with **1.5 seconds of digital silence** at both the beginning and end. Importantly, the final timestamps are adjusted to reflect the original audio, ensuring that this internal processing step *does not affect temporal accuracy*.

### 3. Speaker diarization methodologies

The tool offers two distinct methods for speaker identification, selectable via the `--diarization` argument:

-   **`pyannote` (Default)**: this method (`--diarization pyannote`) uses the pre-trained, end-to-end `pyannote/speaker-diarization-3.1` pipeline [[2]](#2). It is a comprehensive model that performs speech segmentation, embedding extraction, and clustering in a single step, offering high accuracy, especially in conversations with overlapping speech.

-   **`cluster`**: This method (`--diarization cluster`) is a multi-step pipeline:
    1.  **Voice Activity Detection (VAD)**: the audio is first segmented into speech and non-speech regions using the `Silero-VAD` model [[4]](#4).
    2.  **Speaker Embedding**: for each speech segment, a fixed-dimensional vector representation (an embedding, or "voiceprint") is extracted using the `speechbrain/spkrec-ecapa-voxceleb` model, which is based on the ECAPA-TDNN architecture [[3]](#3).
    3.  **Clustering**: all embeddings are grouped using agglomerative clustering based on cosine similarity to identify the unique speakers.

### 4. Intelligent chunking for file processing

When transcribing a file (using the `--file` argument), simply splitting the audio into fixed-size chunks can cut sentences in half and destroy conversational context. So, this script leverages a different approach:

1.  The entire file is first diarized to identify all speaker turns.
2.  In `transcription` mode, these turns are then grouped into optimal chunks of 60-120 seconds.
3.  In `subtitle` mode, the cuts between chunks are made at natural pauses (silences) between speaker segments.

This method ensures that each chunk sent for transcription contains coherent conversational context, improving the accuracy and readability of the output.

### 5. Output formatting: transcription vs. subtitle

The final output format is controlled by the `--mode` argument, which dictates how text segments are presented.

-   **`transcription` Mode**: this mode is designed for maximum readability. It performs a post-processing step that intelligently concatenates consecutive utterances from the same speaker. This transforms a fragmented stream of segments into coherent, paragraph-like blocks of text for each speaker, making the final document easy to read, analyze, and quote.

-   **`subtitle` Mode**: this mode is optimized for real-time or time-sensitive applications. It outputs each transcribed segment immediately with its corresponding timestamp, without concatenation. This ensures that the output stays as close as possible to the live audio stream, which is critical for subtitling or immediate monitoring, even if it results in more frequent timestamps.

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
| `--language <lang>` | `fr` | Language code (e.g., `en`, `es`, `de`, `fr`, `nl`, `pt`, `ro`, etc.). |
| `--model <id>` | `large-v3` | Whisper model [[1]](#1). Choices: `tiny`, `base`, `small`, `medium`, `large`, `large-v1/v2/v3`, and distilled variants : `tiny.en`, `base.en`, `small.en`, `medium.en`, `distil-large-v2`, `distil-medium.en`, `distil-small.en`. |
| `--file <path>` | `None` | Path to a `.wav` file. If omitted, runs in live mode. |
| `--mode <mode>` | `subtitle` | Output mode. Choices: `subtitle`, `transcription`. |
| `--diarization <method>` | `pyannote` | Diarization method. Choices: `pyannote`, `cluster`. |
| `--threshold <float>` | `0.7` | **(Cluster only)** Similarity threshold for speaker clustering (0.0-1.0). |
| `--min-speakers <int>` | `None` | **(File mode only)** Hint for the minimum number of speakers. |
| `--max-speakers <int>` | `None` | **(File mode only)** Hint for the maximum number of speakers. |
| `--transcription-engine <engine>` | `auto` | Engine. Choices: `auto`, `faster-whisper`, `transformers`. |
| `--auto-engine-threshold <float>` | `15.0` | **(Auto mode only)** Time in seconds to switch from `transformers` to `faster-whisper`. |
| `--enhancement <method>` | `none` | Audio enhancement. Choices: `none`, `deepfilternet` [[6]](#6), `demucs` [[5]](#5). |

## Usage scenarios

### Scenario 1: live subtitling of a presentation

**Goal**: low-latency, real-time captions for a single speaker.
**Rationale**: `cluster` is faster than `pyannote`. A lower threshold (`0.4`) prevents voice modulation from creating a new speaker identity. The `small` model is selected for its strong balance of speed and accuracy in English.
```bash
python main.py --mode subtitle --language en --diarization cluster --threshold 0.4 --model small
```

### Scenario 2: generating a transcript of a recorded interview

**Goal**: high-accuracy transcript of a 2-person conversation.
**Rationale**: `pyannote` is highly accurate for diarization. Specifying `--min-speakers 2 --max-speakers 2` constrains the model for optimal results. `transcription` mode creates a clean, readable document. `deepfilternet` provides lightweight noise reduction.
```bash
python main.py --mode transcription --file "interview.wav" --min-speakers 2 --max-speakers 2 --enhancement deepfilternet
```

### Scenario 3: transcribing a noisy field recording

**Goal**: extract intelligible speech from a noisy environment.
**Rationale**: `demucs` is a powerful source separation model that can isolate vocals. The resulting audio may be fragmented, so the `transformers` engine is used for its stability on short segments.
```bash
python main.py --mode transcription --file "field_recording.wav" --enhancement demucs --transcription-engine transformers
```

### Scenario 4: processing a multi-speaker focus group recording

**Goal**: transcribe a complex conversation with an unknown number of speakers.
**Rationale**: `pyannote` excels in complex, multi-speaker scenarios. The default `auto` engine will balance speed and accuracy across varying segment lengths.
```bash
python main.py --mode transcription --file "focus_group.wav" --diarization pyannote
```

## Output file naming

All transcriptions are saved to a `.txt` file with a name generated from the configuration settings.

-   **Format**: `{base_name}_{mode}_{model}_{diarization}_{details}_{timestamp}.txt`
-   **Components**:
    -   `base_name`: the original filename, or `live` for microphone recordings.
    -   `mode`: `subtitle` or `transcription`.
    -   `model`: the Whisper model used (e.g., `large-v3`).
    -   `diarization`: `pyannote` or `cluster`.
    -   `details`: `threshX.XX` for `cluster` mode, or `N-speakers` for file-based `pyannote` mode.
    -   `timestamp`: a `YYYYMMDD_HHMMSS` string added to live recordings.
-   **Examples**:
    -   `interview_transcription_large-v3_pyannote_2-speakers.txt`
    -   `live_subtitle_small_cluster_thresh0.60_20251002_213000.txt`

## Troubleshooting

### Repetitive or inaccurate text

-   **Symptom**: the transcription repeats a phrase or contains nonsensical text.
-   **Cause**: this can occur when short or silent segments are passed to the transcription model.
-   **Solution**:
    1.  Lower `--auto-engine-threshold` (e.g., to `10.0`) to use the `transformers` engine on more segments.
    2.  Force the `transformers` engine for the entire run with `--transcription-engine transformers`.

### Incorrect speaker labels

-   **Symptom**: one person is labeled as multiple speakers, or multiple people are merged.
-   **Solution**:
    1.  If using `--diarization cluster`, adjust the `--threshold`. A lower value merges more easily; a higher value separates more easily.
    2.  If using `--diarization pyannote` on a file, provide speaker hints with `--min-speakers` and `--max-speakers`.

### Slow performance

-   **Symptom**: transcription speed is significantly slower than real-time.
-   **Solution**:
    1.  Ensure the GPU-accelerated version of PyTorch is installed and being utilized.
    2.  Use a smaller model (e.g., `--model medium` or `--model small`).
    3.  For clean audio, force the faster engine with `--transcription-engine faster-whisper`.

## Bibliography

1.  <a name="1"></a>Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
2.  <a name="2"></a>Bredin, H., Yin, R., Coria, J. M., Gelly, G., Korshunov, P., Lavechin, M., Fustes, D., Titeux, H., Bouaziz, W., & Gill, M. (2020). *pyannote.audio: neural building blocks for speaker diarization*. [arXiv:1911.01255](https://arxiv.org/abs/1911.01255).
3.  <a name="3"></a>Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*. [arXiv:2005.07143](https://arxiv.org/abs/2005.07143).
4.  <a name="4"></a>Silero Team. (2024). *Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier*. [GitHub repository](https://github.com/snakers4/silero-vad).
5.  <a name="5"></a>Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2019). *Music Source Separation in the Waveform Domain*. [arXiv:1911.13254](https://arxiv.org/abs/1911.13254).
6.  <a name="6"></a>Schröter, H., Escalante-B., A. N., & Rosenkranz, T. (2022). *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*. [arXiv:2110.05588](https://arxiv.org/abs/2110.05588).

## License

This project is licensed under the MIT License.
