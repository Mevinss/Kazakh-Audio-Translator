import logging
import time

from modules.transcribers.base_transcriber import BaseTranscriber

logger = logging.getLogger(__name__)

# Check if faster_whisper module is available
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    # WhisperModel not used when FASTER_WHISPER_AVAILABLE is False
    # The _load_model method checks the flag before using WhisperModel
    logger.warning(
        "faster-whisper модулі орнатылмаған. "
        "Орнату үшін: pip install faster-whisper"
    )


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using CTranslate2-based Faster-Whisper large-v3 model."""

    MODEL_SIZE = 'large-v3'

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            if not FASTER_WHISPER_AVAILABLE:
                raise ImportError(
                    "faster-whisper модулі орнатылмаған. "
                    "Орнату үшін: pip install faster-whisper"
                )
            logger.info("Loading Faster-Whisper %s model…", self.MODEL_SIZE)
            self._model = WhisperModel(
                self.MODEL_SIZE,
                device='cpu',
                compute_type='int8',
                download_root='models',
            )
            logger.info("Faster-Whisper %s model loaded.", self.MODEL_SIZE)

    def transcribe(self, audio_path: str, language: str = 'kk') -> dict:
        self._load_model()
        start = time.time()
        lang_arg = None if language in (None, 'auto') else language
        segs, info = self._model.transcribe(
            audio_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
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
        detected_lang = getattr(info, 'language', None) or lang_arg or 'kk'

        return {
            'text': ' '.join(s['text'] for s in segments),
            'segments': segments,
            'duration': round(duration, 2),
            'confidence': confidence,
            'processing_time': round(elapsed, 2),
            'language': detected_lang,
        }
