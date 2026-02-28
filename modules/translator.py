"""
Machine translation module for Kazakh Audio Translator.

Provides lightweight EN→KK and RU→KK translation using Helsinki-NLP MarianMT
models (~290 MB each, downloaded from HuggingFace on first use).

This is the machine-translation component of the research project: spoken
audio in English or Russian is first transcribed by an ASR model, then the
resulting text is translated to Kazakh by the MarianTranslator.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Helsinki-NLP model IDs for each (source, target) language pair.
# These are quantised MarianMT models that run comfortably on CPU.
_MODEL_IDS = {
    ('en', 'kk'): 'Helsinki-NLP/opus-mt-en-kk',
    ('ru', 'kk'): 'Helsinki-NLP/opus-mt-ru-kk',
}

# Human-readable names shown in the UI
LANGUAGE_NAMES = {
    'kk': 'Казахский',
    'en': 'Английский',
    'ru': 'Русский',
}


class MarianTranslator:
    """Translates text to Kazakh using Helsinki-NLP MarianMT models.

    Models are loaded lazily on the first call to :meth:`translate` and then
    kept in memory for the lifetime of the process.
    """

    def __init__(self):
        self._cache: dict = {}  # (src, tgt) -> (model, tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str, src: str, tgt: str = 'kk') -> str:
        """Translate *text* from language *src* to *tgt* (default: Kazakh).

        Long texts are split into chunks of ≤ 100 words to stay within the
        model's 512-token context window.

        Returns an empty string if *text* is blank or the pair is unsupported.
        """
        if not text or not text.strip():
            return ''

        key = (src, tgt)
        if key not in _MODEL_IDS:
            logger.warning("Translation pair %s→%s is not supported; skipping.", src, tgt)
            return ''

        model, tokenizer = self._load(key)

        chunks = _split_text(text)
        translated = []
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )
            output_ids = model.generate(**inputs)
            translated.append(
                tokenizer.decode(output_ids[0], skip_special_tokens=True)
            )

        return ' '.join(translated)

    @staticmethod
    def supported_sources(tgt: str = 'kk') -> list:
        """Return source language codes that can be translated to *tgt*."""
        return [src for (src, t) in _MODEL_IDS if t == tgt]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, key: tuple):
        if key in self._cache:
            return self._cache[key]

        model_id = _MODEL_IDS[key]
        logger.info("Loading translation model %s …", model_id)
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id)
        logger.info("Translation model %s loaded.", model_id)
        self._cache[key] = (model, tokenizer)
        return model, tokenizer


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-created on first use)
# ---------------------------------------------------------------------------
_translator: Optional[MarianTranslator] = None


def get_translator() -> MarianTranslator:
    """Return the module-level MarianTranslator singleton."""
    global _translator
    if _translator is None:
        _translator = MarianTranslator()
    return _translator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_text(text: str, max_words: int = 100) -> list:
    """Split *text* into chunks of at most *max_words* words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [
        ' '.join(words[i: i + max_words])
        for i in range(0, len(words), max_words)
    ]
