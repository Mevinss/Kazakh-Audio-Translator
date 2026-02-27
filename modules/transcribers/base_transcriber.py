from abc import ABC, abstractmethod


class BaseTranscriber(ABC):
    """Abstract base class for all ASR transcribers."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file.

        Returns a dict with keys:
            text        (str)   – transcribed text
            segments    (list)  – list of segment dicts with start/end/text
            duration    (float) – audio duration in seconds
            confidence  (float) – mean confidence score 0-1
        """

    def _model_name(self) -> str:
        return self.__class__.__name__
