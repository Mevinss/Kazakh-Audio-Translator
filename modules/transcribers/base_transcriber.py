from abc import ABC, abstractmethod


class BaseTranscriber(ABC):
    """Abstract base class for all ASR transcribers."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: str = 'kk') -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to the audio file (WAV, 16 kHz mono recommended).
            language:   BCP-47 language code of the spoken language, e.g. 'kk'
                        for Kazakh, 'en' for English, 'ru' for Russian.
                        Pass 'auto' (or None) to let the model auto-detect.

        Returns a dict with keys:
            text        (str)   – transcribed text
            segments    (list)  – list of segment dicts with start/end/text
            duration    (float) – audio duration in seconds
            confidence  (float) – mean confidence score 0-1
            language    (str)   – detected/used language code
        """

    def _model_name(self) -> str:
        return self.__class__.__name__
