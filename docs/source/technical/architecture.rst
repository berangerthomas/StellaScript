###################################
StellaScript Project Architecture
###################################

This document details the structure of the StellaScript project, the role of each file, and how the modules interact to perform audio transcription and diarization.

Overview
========

The project is structured around a main module, ``stellascript``, which contains all the application logic. Execution is initiated by ``main.py`` at the project root, which acts as the entry point.

Root Files
==========

-   ``main.py``: **Application entry point.** It is responsible for parsing command-line arguments, initializing the orchestrator, and launching the transcription process (either live or from a file).
-   ``README.md``: **Main documentation.** Provides an overview of the project, installation instructions, and usage guidelines.
-   ``pyproject.toml`` & ``uv.lock``: **Dependency management.** These files define the Python libraries required for the project to function.
-   ``.gitignore``: Configuration file for Git, specifying files and folders to be ignored.
-   ``LICENSE``: Contains the MIT license under which the project is distributed.

``stellascript`` Module (Application Core)
============================================

The ``stellascript/`` directory contains the main source code of the application, organized into several modules and sub-modules.

-   ``orchestrator.py``: **The conductor.** This is the most important file in the project. The ``StellaScriptTranscription`` class manages the entire processing pipeline. It initializes the various components (transcriber, diarizer, etc.) and coordinates their interactions, whether for real-time or file-based processing.
-   ``config.py``: **Central configuration.** This file centralizes all technical constants and parameters used in the application (e.g., sampling rate, audio buffer duration, voice detection thresholds). This allows for easy modification of the application's behavior from a single location.
-   ``cli.py``: **Command-line interface.** Defines all the arguments that the user can pass to the program (such as ``--file``, ``--language``, ``--mode``) and ensures they are correctly interpreted.
-   ``logging_config.py``: **Logging configuration.** Sets up the logging system to display informational messages, warnings, or errors during execution, which is crucial for debugging.

``stellascript/audio`` Sub-module
------------------------------------

This module is dedicated to handling raw audio data.

-   ``capture.py``: **Audio capture.** Manages interaction with the microphone to record the audio stream in real-time.
-   ``enhancement.py``: **Audio enhancement.** Contains the logic for applying audio cleaning models, such as ``DeepFilterNet`` or ``Demucs``, to reduce background noise and improve voice clarity before transcription.

``stellascript/processing`` Sub-module
-----------------------------------------

This module contains the components responsible for the intelligent analysis and processing of audio.

-   ``transcriber.py``: **Transcription module.** Encapsulates the speech recognition model (Whisper via ``whisperx``). Its sole responsibility is to take an audio segment and convert it into text.
-   ``diarizer.py``: **Diarization module.** Its role is to answer the question: "who is speaking and when?". It uses models like ``pyannote.audio`` or a combination of VAD (Voice Activity Detection) and clustering to segment the audio based on speakers.
-   ``speaker_manager.py``: **Speaker manager.** Works closely with the ``diarizer``, especially for the ``cluster`` method. It is responsible for creating and managing "voiceprints" (embeddings) to identify and differentiate speakers consistently.
