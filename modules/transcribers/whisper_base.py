import logging
import time

from modules.transcribers.base_transcriber import BaseTranscriber

logger = logging.getLogger(__name__)

# Kazakh initial prompt helps the model produce Kazakh-specific characters
_KK_INITIAL_PROMPT = (
    "Қазақстан, сәлеметсіз бе, қалайсыз, рахмет, жақсы, "
    "әлем, өмір, үкімет, ұлт, білім, ғылым, іс-шара."
)


class WhisperBaseTranscriber(BaseTranscriber):
    """Transcriber using OpenAI Whisper base model."""

    MODEL_SIZE = 'base'

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import whisper
            except ImportError as e:
                logger.error(
                    "Failed to import 'whisper'. Make sure 'openai-whisper' is installed: "
                    "pip install openai-whisper. Original error: %s", e
                )
                raise ImportError(
                    "Модуль 'whisper' табылмады. 'openai-whisper' орнатыңыз: "
                    "pip install openai-whisper"
                ) from e
            logger.info("Loading Whisper %s model…", self.MODEL_SIZE)
            self._model = whisper.load_model(self.MODEL_SIZE)
            logger.info("Whisper %s model loaded.", self.MODEL_SIZE)

    def transcribe(self, audio_path: str, language: str = 'kk') -> dict:
        self._load_model()
        start = time.time()
        # Pass None for auto-detect; otherwise use the specified language code.
        lang_arg = None if language in (None, 'auto') else language
        result = self._model.transcribe(
            audio_path,
            language=lang_arg,
            task='transcribe',
            verbose=False,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4),
            condition_on_previous_text=True,
            initial_prompt=_KK_INITIAL_PROMPT if lang_arg == 'kk' else None,
            word_timestamps=True,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.4,
            logprob_threshold=-0.8,
        )
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
            'language': result.get('language', lang_arg or 'kk'),
        }
