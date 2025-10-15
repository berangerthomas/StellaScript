.. _concepts_whisper_models:

#######################
Whisper Model Selection
#######################

The choice of the Whisper model is a trade-off between transcription accuracy (measured by Word Error Rate, or WER) and processing speed. The graph below shows the performance of different Whisper models across four languages, which can help in selecting the most appropriate model for a given task.

.. figure:: /pictures/choosing_whisper_model_wer_vs_time.png
   :alt: Whisper Model Performance

   Word Error Rate (WER) versus processing time for various Whisper models in French, English, Spanish, and German.

Based on the performance data, here are some model selection recommendations for specific languages:

-   **French**: ``large-v3`` is recommended for the highest accuracy.
-   **English**: the ``small`` model performs slightly better than ``large-v3`` while being nearly four times faster.
-   **Spanish**: the ``medium`` model provides a balance between accuracy and processing speed.
-   **German**: ``large-v3`` achieves the lowest error rate among the tested models.
