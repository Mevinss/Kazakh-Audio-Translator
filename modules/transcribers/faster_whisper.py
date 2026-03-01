import logging
import time

from modules.transcribers.base_transcriber import BaseTranscriber

logger = logging.getLogger(__name__)

# Kazakh initial prompt helps the model produce Kazakh-specific characters
_KK_INITIAL_PROMPT = (
    "Қазақстан, сәлеметсіз бе, қалайсыз, рахмет, жақсы, "
    "әлем, өмір, үкімет, ұлт, білім, ғылым, іс-шара."
)


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using CTranslate2-based Faster-Whisper large-v3 model."""

    MODEL_SIZE = 'large-v3'

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                logger.error(
                    "Failed to import 'faster_whisper'. Make sure 'faster-whisper' is installed: "
                    "pip install faster-whisper. Original error: %s", e
                )
                raise ImportError(
                    "Модуль 'faster_whisper' табылмады. 'faster-whisper' орнатыңыз: "
                    "pip install faster-whisper"
                ) from e
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
            task='transcribe',
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4),
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
            condition_on_previous_text=True,
            initial_prompt=_KK_INITIAL_PROMPT if lang_arg == 'kk' else None,
            word_timestamps=True,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-0.8,
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
