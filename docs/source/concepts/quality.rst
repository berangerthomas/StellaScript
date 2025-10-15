.. _quality:

###########################################################
Optimizing Transcription Quality with Whisper
###########################################################

The transcription performance of Whisper models can be improved by a combination of audio preprocessing, model optimizations, and post-processing techniques. Published research indicates that certain methods can reduce the Word Error Rate (WER) while maintaining processing speed.

.. contents::
   :local:

=======================================
Audio Preprocessing
=======================================

----------------------------------
Voice Activity Detection (VAD)
----------------------------------

Voice Activity Detection is a technique used for improving Whisper transcription. By identifying segments containing speech, VAD can help eliminate silent or noisy sections that may cause hallucinations [1]_ [2]_ [3]_ [4]_ [5]_ [6]_.

**Reported Benefits:**

*   **45% reduction in transcription errors**, according to several studies [4]_ [1]_.
*   Elimination of spurious transcriptions in non-vocal segments [3]_ [7]_.
*   Significant accuracy improvement on telephone recordings [4]_.
*   Reduced computational load by avoiding the processing of unnecessary segments [8]_ [1]_.

Recommended VAD models include Silero-VAD and WebRTC VAD, with Silero-VAD showing superior performance on complex data [2]_ [6]_.

----------------------------------
Neural Denoising
----------------------------------

Deep learning-based source separation techniques, such as **Demucs** from Meta, can be used for this purpose. This approach uses multi-layer convolutional neural networks to separate speech from background noise [9]_ [10]_.

**Documented Results:**

*   Demucs, followed by a low-pass filter, **reduced the performance gap between genres by 25%** for Whisper [10]_.
*   Particularly notable improvements on recordings with significant ambient noise [9]_.
*   Superior performance compared to traditional signal-based denoising techniques [10]_.

------------------------------------------
Audio Normalization and Parameters
------------------------------------------

Whisper models are pre-trained with a 16 kHz sampling rate. **Optimizing audio parameters** can yield benefits [11]_ [12]_ [13]_:

*   **Recommended Format**: MP3 mono at 16 kbps and 12-16 kHz [14]_.
*   **Latency Reduction**: Up to 50% without loss of accuracy [14]_.
*   **Audio Level Normalization**: Improves transcription consistency [15]_.

================================
Model Fine-Tuning and Adaptation
================================

-----------------------------------------------
Fine-Tuning with LoRA (Low-Rank Adaptation)
-----------------------------------------------

Fine-tuning using LoRA is a technique for enhancing Whisper's performance on specific domains [16]_ [17]_ [18]_ [19]_.

**Reported Performance:**

*   **WER reduction from 68.49% to 26.26%** (a 61.7% improvement) on aeronautical data [16]_.
*   Uses only **0.8% of the model's parameters** for fine-tuning [17]_ [16]_.
*   **38.49% WER improvement** on Vietnamese with Whisper-Tiny [18]_.
*   Maintains generalization on data not seen during training [17]_.

**Identified Hyperparameters:**

*   **Learning Rate**: 1e-3 for Large, 1e-5 for Turbo [16]_ [17]_.
*   **LoRA Alpha**: 256 for best performance [17]_.
*   **LoRA Rank**: 32 as a starting point [17]_.

----------------------------------
Transcription Normalization
----------------------------------

Text normalization schemes can improve evaluation metrics. OpenAI provides a specialized normalizer that [16]_ [20]_:

*   Standardizes case and removes punctuation [20]_.
*   Handles regional spelling variations [20]_.
*   Improves WER scores by an average of 1.78% [21]_.

=====================================
Model Optimization and Acceleration
=====================================

----------------------------------
CTranslate2 and faster-whisper
----------------------------------

The CTranslate2 implementation (`faster-whisper`) is a common method for performance optimization [22]_ [23]_ [24]_.

**Measured Improvements:**

*   **Speed**: Up to **4x faster** than the original implementation [23]_ [22]_.
*   **Memory**: VRAM usage reduced from 11.3 GB to 4.7 GB for Large-v2 [23]_.
*   **Quantization**: Further reduction to 3.1 GB with INT8 [23]_.
*   **Accuracy Maintained**: Performance is identical to the original [22]_.

----------------------------------
Quantization
----------------------------------

Quantization techniques enable deployment on resource-constrained hardware [25]_ [26]_.

*   **INT8 Quantization**: 19% latency reduction, 45% size reduction [25]_.
*   **Accuracy Maintained**: 98.4% accuracy with INT4 [25]_.
*   **Automatic Optimization**: CTranslate2 handles quantization transparently [22]_.

========================================
Segmentation and Decoding Strategies
========================================

----------------------------------
Audio Segmentation
----------------------------------

The audio chunking strategy influences transcription quality [27]_ [28]_ [29]_.

**Recommended Approaches:**

*   **VAD-based Segmentation**: Splitting at natural speech boundaries [30]_ [27]_.
*   **Overlap**: 10-20% overlap between segments [31]_ [32]_.
*   **Chunk Size**: 1-second segments with attention-guided stopping [28]_.

----------------------------------
Decoding Parameter Optimization
----------------------------------

Decoding parameters have a significant impact on quality [33]_ [34]_ [35]_ [36]_.

**Identified Configuration:**

*   **Beam Size**: 5 provides a balance between quality and speed [34]_ [35]_ [33]_.
*   **Temperature**: 0.0 to maximize consistency [35]_ [34]_.
*   **Language Setting**: Explicitly specifying the language improves performance by up to 10x [37]_ [34]_.
*   **`condition_on_previous_text`**: `False` to prevent hallucinatory loops [38]_ [33]_.

========================
Hallucination Prevention
========================

-------------------------------------------
Detection and Prevention Techniques
-------------------------------------------

Hallucinations can be a challenge, especially in non-vocal segments [39]_ [5]_ [6]_.

**Proposed Solutions:**

*   **Calm-Whisper**: Selectively fine-tuning 3 attention heads reduces hallucinations by 80% [5]_.
*   **Bag of Hallucinations (BoH)**: Detects and suppresses recurring phrases [6]_.
*   **Adaptive Thresholds**: `compression_ratio_threshold` and `log_prob_threshold` [36]_.
*   **Post-processing**: Aho-Corasick algorithm for pattern detection [6]_.

----------------------------------
Anti-Hallucination Parameters
----------------------------------

Recommended configuration to minimize hallucinations [33]_ [36]_:

*   `no_speech_threshold`: Adjust according to desired sensitivity.
*   `compression_ratio_threshold`: 2.4 by default.
*   `log_prob_threshold`: -1.0 to filter uncertain transcriptions.

===========================================
Evaluation and Benchmarking Methodologies
===========================================

----------------------------------
Standardized Metrics
----------------------------------

Rigorous evaluation requires proper normalization [21]_ [40]_.

*   **Normalized WER**: Use the OpenAI normalizer [20]_ [21]_.
*   **Realistic Datasets**: Prefer "in-the-wild" data over academic corpora [40]_.
*   **Multilingual Consistency**: Use language-specific normalization [40]_.

----------------------------------
Deployment Considerations
----------------------------------

Studies show that real-world performance can differ from academic benchmarks. The FLEURS dataset, for example, may overestimate performance compared to natural recordings [40]_.

================================
Summary of Methods
================================

An integrated strategy combines several complementary approaches:

1.  **Preprocessing**: VAD + Demucs for denoising + audio normalization.
2.  **Model**: `faster-whisper` with INT8 quantization + LoRA fine-tuning for specific domains.
3.  **Decoding**: `beam_size=5`, `temperature=0`, language specification.
4.  **Post-processing**: Text normalization + hallucination detection.

This combined approach can reduce WER, as demonstrated in aeronautical and multilingual case studies.

.. rubric:: References

.. [1] https://www.osedea.com/insight/understanding-voice-activity-detection-how-vad-powers-real-time-voice-systems
.. [2] https://github.com/openai/whisper/discussions/2378
.. [3] https://www.f22labs.com/blogs/what-is-vad-and-diarization-with-whisper-models-a-complete-guide/
.. [4] https://docs.phonexia.com/products/speech-platform-4/3.2.0/technologies/speech-to-text/enhanced-speech-to-text-built-on-whisper/comparison
.. [5] https://arxiv.org/html/2505.12969v1
.. [6] https://arxiv.org/html/2501.11378v1
.. [7] https://aclanthology.org/2025.iwsds-1.26.pdf
.. [8] https://arxiv.org/html/2506.01365v1
.. [9] https://github.com/openai/whisper/discussions/2125
.. [10] https://arxiv.org/html/2410.16712v1
.. [11] https://learnopencv.com/fine-tuning-whisper-on-custom-dataset/
.. [12] https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b
.. [13] https://github.com/openai/whisper/discussions/870
.. [14] https://dev.to/mxro/optimise-openai-whisper-api-audio-format-sampling-rate-and-quality-29fj
.. [15] https://myscale.com/blog/mastering-audio-transcription-with-whisper-ai-step-by-step-guide/
.. [16] https://arxiv.org/html/2506.21990v1
.. [17] https://arxiv.org/pdf/2503.22692.pdf
.. [18] https://trellisdata.com/research/Blog%20Post%20Title%20One-crc24-m7skl
.. [19] https://github.com/Vaibhavs10/fast-whisper-finetuning
.. [20] https://huggingface.co/learn/audio-course/chapter5/evaluation
.. [21] https://mlcommons.org/2025/09/whisper-inferencev5-1/
.. [22] https://github.com/SYSTRAN/faster-whisper
.. [23] https://nikolas.blog/making-openai-whisper-faster/
.. [24] https://ai.gopubby.com/whisper-gets-a-boost-introducing-fast-whisper-506f1901a8b2
.. [25] https://arxiv.org/html/2503.09905v1
.. [26] https://arxiv.org/pdf/2503.09905.pdf
.. [27] https://github.com/openai/whisper/discussions/1977
.. [28] https://arxiv.org/pdf/2406.10052.pdf
.. [29] https://community.groq.com/t/chunking-longer-audio-files-for-whisper-models-on-groq/162
.. [30] https://www.cerebrium.ai/articles/faster-whisper-transcription-how-to-maximize-performance-for-real-time-audio-to-text
.. [31] https://weaviate.io/blog/chunking-strategies-for-rag
.. [32] https://huggingface.co/openai/whisper-large-v2/discussions/67
.. [33] https://github.com/jhj0517/Whisper-WebUI/wiki/Whisper-Advanced-Parameters
.. [34] https://arxiv.org/html/2503.23542v1
.. [35] https.
