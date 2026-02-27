import logging
import time

from modules.transcribers.base_transcriber import BaseTranscriber

logger = logging.getLogger(__name__)


class WhisperMediumTranscriber(BaseTranscriber):
    """Transcriber using OpenAI Whisper medium model."""

    MODEL_SIZE = 'medium'

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper
            logger.info("Loading Whisper %s model…", self.MODEL_SIZE)
            self._model = whisper.load_model(self.MODEL_SIZE)
            logger.info("Whisper %s model loaded.", self.MODEL_SIZE)

    def transcribe(self, audio_path: str) -> dict:
        self._load_model()
        start = time.time()
        result = self._model.transcribe(audio_path, language='kk', verbose=False)
        elapsed = time.time() - start

        segments = [
            {'start': s['start'], 'end': s['end'], 'text': s['text'].strip()}
            for s in result.get('segments', [])
        ]

        # avg_logprob is negative (e.g. -0.2 → 80% confidence); clamp to [0, 1]
        confidences = [
            s.get('avg_logprob', -1.0)
            for s in result.get('segments', [])
            if 'avg_logprob' in s
        ]
        if confidences:
            confidence = round(max(0.0, min(1.0, 1.0 + (sum(confidences) / len(confidences)))), 4)
        else:
            confidence = 0.0

        duration = result.get('segments', [{}])[-1].get('end', elapsed) if segments else elapsed

        return {
            'text': result.get('text', '').strip(),
            'segments': segments,
            'duration': round(duration, 2),
            'confidence': confidence,
            'processing_time': round(elapsed, 2),
        }
