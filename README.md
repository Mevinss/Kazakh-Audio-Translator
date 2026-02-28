# 🎙️ Kazakh ASR Comparison

Простое веб-приложение для транскрибирования казахского аудио/видео и сравнения трёх ASR-моделей: **Whisper Base**, **Whisper Medium** и **Faster-Whisper Large-v3**.

---

## 📋 Системные требования

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) (для извлечения аудио из видео)
- 4+ ГБ ОЗУ (для Faster-Whisper Large-v3)
- CUDA GPU (опционально, ускоряет транскрибирование)

---

## 🚀 Установка и запуск

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/Mevinss/Kazakh-Audio-Translator.git
cd Kazakh-Audio-Translator

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows

# 3. Установите зависимости
pip install -r requirements.txt

# 4. (Linux/macOS) Установите FFmpeg
sudo apt install ffmpeg          # Debian/Ubuntu
# brew install ffmpeg            # macOS

# 5. Запустите приложение
python app.py
# Для режима отладки: FLASK_DEBUG=1 python app.py
```

Откройте [http://localhost:5000](http://localhost:5000) в браузере.

---

## 🗂️ Структура проекта

```
Kazakh-Audio-Translator/
├── app.py                          # Главное Flask приложение
├── config.py                       # Конфигурация
├── requirements.txt
├── README.md
├── .gitignore
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   ├── index.html                  # Главная страница (загрузка файла)
│   ├── results.html                # Страница результатов и метрик
│   └── history.html                # История транскрибирований
├── modules/
│   ├── audio_processor.py          # Извлечение и нормализация аудио
│   ├── metrics.py                  # WER / CER
│   ├── database.py                 # SQLite (история)
│   └── transcribers/
│       ├── base_transcriber.py     # Абстрактный базовый класс
│       ├── whisper_base.py         # Whisper Base (74 M параметров)
│       ├── whisper_medium.py       # Whisper Medium (307 M)
│       └── faster_whisper.py       # Faster-Whisper Large-v3
├── uploads/                        # Загруженные файлы (авто-очистка)
├── models/                         # Кэш моделей
└── tests/
    └── test_transcribers.py        # Unit- и интеграционные тесты
```

---

## 🖥️ Использование интерфейса

1. **Главная страница** — перетащите аудио/видео файл в зону загрузки или нажмите на неё.
2. Выберите одну или несколько **ASR-моделей** для сравнения.
3. Опционально введите **референсный текст** для расчёта WER/CER.
4. Нажмите **«Транскрибировать»**.
5. На **странице результатов** отображается:
   - Сравнительная таблица (WER, CER, Confidence, Время обработки)
   - Транскрипция каждой модели с таймстемпами
   - Кнопка экспорта в CSV
6. **История** хранится в SQLite и доступна по кнопке «История».

---

## 📊 Метрики

| Метрика | Описание | Хорошее значение |
|---------|----------|-----------------|
| **WER** | Word Error Rate — доля слов с ошибками | < 15% |
| **CER** | Character Error Rate — доля символов с ошибками | < 10% |
| **Confidence** | Средняя уверенность модели (0–100%) | > 70% |

> WER и CER рассчитываются только при наличии референсного текста.

---

## ➕ Как добавить новую модель

1. Создайте файл `modules/transcribers/my_model.py`:

```python
from modules.transcribers.base_transcriber import BaseTranscriber

class MyModelTranscriber(BaseTranscriber):
    def transcribe(self, audio_path: str) -> dict:
        # ... загрузка и запуск модели ...
        return {
            'text': '...',
            'segments': [],
            'duration': 0.0,
            'confidence': 0.0,
            'processing_time': 0.0,
        }
```

2. Зарегистрируйте модель в `config.py`:

```python
MODEL_MY_MODEL = 'my_model'
MODEL_DISPLAY_NAMES[MODEL_MY_MODEL] = 'Моя модель'
```

3. Добавьте ветку в функцию `_get_transcriber()` в `app.py`.

---

## 🧪 Запуск тестов

```bash
pip install jiwer flask werkzeug   # минимальные зависимости для тестов
python -m pytest tests/ -v
```

---

## 📄 Лицензия

MIT
