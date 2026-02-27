import logging
import time

from modules.transcribers.base_transcriber import BaseTranscriber

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using CTranslate2-based Faster-Whisper large-v3 model."""

    MODEL_SIZE = 'large-v3'

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading Faster-Whisper %s model…", self.MODEL_SIZE)
            self._model = WhisperModel(
                self.MODEL_SIZE,
                device='cpu',
                compute_type='int8',
                download_root='models',
            )
            logger.info("Faster-Whisper %s model loaded.", self.MODEL_SIZE)

    def transcribe(self, audio_path: str) -> dict:
        self._load_model()
        start = time.time()
        segs, info = self._model.transcribe(
            audio_path,
            language='kk',
            beam_size=5,
            vad_filter=True,
        )
        segs = list(segs)  # consume the generator
        elapsed = time.time() - start

        segments = [
            {'start': s.start, 'end': s.end, 'text': s.text.strip()}
            for s in segs
        ]

        confidences = [
            s.avg_logprob for s in segs if hasattr(s, 'avg_logprob')
        ]
        if confidences:
            confidence = round(max(0.0, min(1.0, 1.0 + (sum(confidences) / len(confidences)))), 4)
        else:
            confidence = 0.0

        duration = info.duration if hasattr(info, 'duration') else elapsed

        return {
            'text': ' '.join(s['text'] for s in segments),
            'segments': segments,
            'duration': round(duration, 2),
            'confidence': confidence,
            'processing_time': round(elapsed, 2),
        }
