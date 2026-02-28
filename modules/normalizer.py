import os
import re
import logging

logger = logging.getLogger(__name__)

# Қазақ тіліндегі дауысты дыбыстар (жуан және жіңішке)
KAZAKH_BACK_VOWELS = set('аоұы')  # Жуан дауыстылар
KAZAKH_FRONT_VOWELS = set('әөүі')  # Жіңішке дауыстылар
KAZAKH_NEUTRAL_VOWELS = set('еиу')  # Бейтарап дауыстылар
KAZAKH_ALL_VOWELS = KAZAKH_BACK_VOWELS | KAZAKH_FRONT_VOWELS | KAZAKH_NEUTRAL_VOWELS

# Қазақ тіліндегі арнайы әріптер
KAZAKH_SPECIAL_CHARS = set('әғқңөұүһі')


class KazakhNormalizer:
    """Қазақ мәтінін нормализациялау - грамматика, синтаксис, морфология түзету."""

    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model_name = os.environ.get("NORMALIZER_MODEL", "gpt-4o-mini")

    def normalize(self, text: str, use_llm: bool = True) -> str:
        """Мәтінді нормализациялау - алдымен LLM, болмаса ереже негізінде."""
        if not text or not text.strip():
            return ""
        if use_llm and self.api_key:
            normalized = self._normalize_with_llm(text)
            if normalized:
                return normalized
        return self._rule_based_normalize(text)

    def _normalize_with_llm(self, text: str) -> str:
        """LLM арқылы мәтінді нормализациялау."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            prompt = (
                "Мына қазақ тілді транскрипцияны әдеби нормаға келтір:\n"
                "1. Тыныс белгілерін дұрыс қой (нүкте, үтір, сұрау белгісі, леп белгісі)\n"
                "2. Сөз жазылуындағы қателерді түзет\n"
                "3. Сингармонизм (үндестік заңын) сақта - жуан/жіңішке дыбыстар үйлесімі\n"
                "4. Грамматикалық жалғауларды тексер\n"
                "5. Сөйлем құрылымын дұрыстап, синтаксисті реттей\n"
                "6. Бас әріптерді дұрыс қолдан\n\n"
                "Тек түзетілген мәтінді қайтар, ешқандай түсініктеме жазба.\n\n"
                f"Транскрипция:\n{text}"
            )
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Сен қазақ тілінің кәсіби редакторысың. "
                            "Қазақ тілінің грамматикасын, синтаксисін, морфологиясын жетік білесің. "
                            "Сингармонизм (үндестік заңы), буын үндестігі туралы білімің бар."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM нормализация сәтсіз, ережелерге негізделген режимге көшу: %s", exc)
            return ""

    def _rule_based_normalize(self, text: str) -> str:
        """Ережелерге негізделген нормализация."""
        if not text:
            return ""

        # 1. Артық бос орындарды жою
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        # 2. Тыныс белгілерінен кейін бос орын қою
        cleaned = re.sub(r"([,.!?;:])([^\s\d])", r"\1 \2", cleaned)
        
        # 3. Тыныс белгілерінен бұрын артық бос орындарды жою
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        
        # 4. Қайталанған тыныс белгілерін жою
        cleaned = re.sub(r"([.!?])\1+", r"\1", cleaned)
        cleaned = re.sub(r",+", ",", cleaned)

        # 5. Сөйлемдерді бөліп, әрқайсысын бас әріппен бастау
        sentences = re.split(r'([.!?])\s*', cleaned)
        result_parts = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            if sentence:
                # Бас әріппен бастау
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                result_parts.append(sentence)
            # Тыныс белгісін қосу
            if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                result_parts.append(sentences[i + 1])
                i += 2
            else:
                i += 1

        cleaned = ''.join(result_parts)

        # 6. Соңында тыныс белгісі болмаса, нүкте қою
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."

        # 7. Бос орындарды қалыпқа келтіру
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _check_vowel_harmony(self, word: str) -> bool:
        """Сингармонизмді (үндестік заңын) тексеру."""
        word_lower = word.lower()
        has_back = any(v in word_lower for v in KAZAKH_BACK_VOWELS)
        has_front = any(v in word_lower for v in KAZAKH_FRONT_VOWELS)
        # Жуан және жіңішке дауыстылар бір сөзде болмауы керек
        return not (has_back and has_front)
