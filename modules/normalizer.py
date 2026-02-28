import os
import re
import logging

logger = logging.getLogger(__name__)

# Common Kazakh vowels used for vowel harmony checks
_HARD_VOWELS = set("аоұыАОҰЫ")
_SOFT_VOWELS = set("әөүіеЄӘӨҮІЕ")
_ALL_VOWELS = _HARD_VOWELS | _SOFT_VOWELS

# Common ASR spelling corrections for Kazakh
_SPELLING_CORRECTIONS = {
    "салеметсиз": "сәлеметсіз",
    "салем": "сәлем",
    "кандай": "қандай",
    "калай": "қалай",
    "калайсын": "қалайсың",
    "калайсыз": "қалайсыз",
    "рахмет": "рахмет",
    "жаксы": "жақсы",
    "коп": "көп",
    "кунге": "күнге",
    "кун": "күн",
    "тусиниктеме": "түсініктеме",
    "казакстан": "қазақстан",
    "казак": "қазақ",
    "бирак": "бірақ",
    "гой": "ғой",
    "кой": "қой",
    "мен": "мен",
    "сен": "сен",
    "ол": "ол",
    "биз": "біз",
    "сиз": "сіз",
    "олар": "олар",
    "болады": "болады",
    "керек": "керек",
    "мумкин": "мүмкін",
    "уакыт": "уақыт",
    "жумыс": "жұмыс",
    "билим": "білім",
    "тил": "тіл",
    "адам": "адам",
    "бала": "бала",
    "уй": "үй",
}


class KazakhNormalizer:
    """Kazakh text normalizer with rule-based grammar/morphology corrections
    and optional LLM enhancement."""

    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model_name = os.environ.get("NORMALIZER_MODEL", "gpt-4o-mini")

    def normalize(self, text: str, use_llm: bool = True) -> str:
        if not text or not text.strip():
            return ""
        if use_llm and self.api_key:
            normalized = self._normalize_with_llm(text)
            if normalized:
                return normalized
        return self._rule_based_normalize(text)

    def _normalize_with_llm(self, text: str) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            prompt = (
                "Мына қазақ мәтінді әдеби нормаға келтір: тыныс белгілерін қой, "
                "грамматиканы түзет, үндестік заңын сақта, морфологиялық "
                "қателерді түзет. Тек түзетілген мәтінді қайтар.\n\n"
                f"Мәтін:\n{text}"
            )
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Сен қазақ тілінің кәсіби редакторысың."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM normalizer fallback to rule-based mode: %s", exc)
            return ""

    def _rule_based_normalize(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        # Apply spelling corrections for common ASR mistakes
        cleaned = self._fix_spelling(cleaned)

        # Fix repeated words (ASR stuttering artifact)
        cleaned = self._fix_repeated_words(cleaned)

        # Fix punctuation spacing
        cleaned = self._fix_punctuation(cleaned)

        # Capitalize first letter of each sentence
        cleaned = self._capitalize_sentences(cleaned)

        # Ensure text ends with terminal punctuation
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."

        return cleaned

    @staticmethod
    def _fix_spelling(text: str) -> str:
        """Apply common ASR transcription error corrections for Kazakh."""
        words = text.split()
        corrected = []
        for word in words:
            if not word:
                continue
            lower = word.lower()
            if lower in _SPELLING_CORRECTIONS:
                replacement = _SPELLING_CORRECTIONS[lower]
                # Preserve original capitalization if the word was capitalized
                if word[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                corrected.append(replacement)
            else:
                corrected.append(word)
        return " ".join(corrected)

    @staticmethod
    def _fix_repeated_words(text: str) -> str:
        """Remove consecutive duplicate words (common ASR artifact)."""
        words = text.split()
        if not words:
            return text
        result = [words[0]]
        for word in words[1:]:
            if word.lower() != result[-1].lower():
                result.append(word)
        return " ".join(result)

    @staticmethod
    def _fix_punctuation(text: str) -> str:
        """Fix common punctuation issues."""
        # Remove space before punctuation marks
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        # Ensure space after punctuation (except at end)
        text = re.sub(r"([.,!?;:])([^\s\d])", r"\1 \2", text)
        # Remove multiple punctuation marks
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        return text

    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        """Capitalize the first letter of each sentence."""
        if not text:
            return text
        # Capitalize first character
        result = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        # Capitalize after sentence-ending punctuation
        result = re.sub(
            r"([.!?]\s+)(\w)",
            lambda m: m.group(1) + m.group(2).upper(),
            result,
        )
        return result
