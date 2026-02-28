import os
import re
import logging

logger = logging.getLogger(__name__)


class KazakhNormalizer:
    """Kazakh text normalizer with optional LLM enhancement."""

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
                "грамматиканы түзет, үндестік заңын сақта. Тек түзетілген мәтінді қайтар.\n\n"
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
        if len(cleaned) > 0:
            cleaned = cleaned[:1].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned
