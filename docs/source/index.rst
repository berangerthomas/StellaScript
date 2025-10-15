.. StellaScript documentation master file.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Introduction

   self

##########################
StellaScript Documentation
##########################

StellaScript is a Python application designed for audio transcription and speaker diarization. Its primary goal is to provide an accurate and efficient tool for converting audio streams, whether pre-recorded or captured live, into structured text while identifying the different speakers.

The system is based on a modular architecture and integrates several state-of-the-art machine learning models for its key features:

*   **Speech Recognition**: Utilizes OpenAI's Whisper model, through the `whisperx` library for optimized performance, to ensure accurate transcription.
*   **Speaker Diarization**: Integrates the `pyannote.audio` pipeline for audio segmentation and turn-taking identification.
*   **Speaker Identification**: Generates voice embeddings with `SpeechBrain` to differentiate and track speakers consistently.

This documentation aims to provide a technical overview of the project, its architecture, and its API.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   concepts/index
   technical/index
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
