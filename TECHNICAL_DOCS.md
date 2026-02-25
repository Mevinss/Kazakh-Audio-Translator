# 🔧 Техническая документация проекта

## 📐 Архитектура системы

### Компоненты

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                    │
│  - React 18, TypeScript                                  │
│  - Tailwind CSS, Zustand                                 │
│  - React Query, Socket.io                                │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP/REST + WebSocket
┌─────────────────▼───────────────────────────────────────┐
│                BACKEND API (FastAPI)                     │
│  - Python 3.11, FastAPI 0.109                           │
│  - JWT Auth, PostgreSQL, Redis                          │
│  - Celery for async tasks                               │
└─────────────────┬───────────────────────────────────────┘
                  │ Task Queue
┌─────────────────▼───────────────────────────────────────┐
│               ML PIPELINE (Python)                       │
│  1. Audio Extraction (FFmpeg)                           │
│  2. ASR (Faster-Whisper large-v3)                      │
│  3. Text Normalization (GPT-4o/Claude/Gemini)          │
│  4. Translation (Multi-model)                           │
│  5. Subtitle Generation (SRT/VTT/ASS)                  │
└─────────────────┬───────────────────────────────────────┘
                  │ Storage
┌─────────────────▼───────────────────────────────────────┐
│         DATA LAYER                                       │
│  - PostgreSQL (метаданные, пользователи)                │
│  - Redis (кэш, очереди)                                 │
│  - MinIO/S3 (файлы, медиа)                             │
└─────────────────────────────────────────────────────────┘
```

---

## 🗄️ База данных (PostgreSQL)

### Схема таблиц

```sql
-- Пользователи
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Файлы
CREATE TABLE files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    filename VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),
    storage_path VARCHAR(1000),
    duration_seconds FLOAT,
    status VARCHAR(50) DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Задачи обработки
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    file_id UUID REFERENCES files(id),
    status VARCHAR(50) DEFAULT 'pending',
    quality VARCHAR(20) DEFAULT 'standard',
    target_lang VARCHAR(5) DEFAULT 'ru',
    progress INT DEFAULT 0,
    current_stage VARCHAR(100),
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Результаты
CREATE TABLE results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id),
    original_text TEXT,
    translated_text TEXT,
    segments JSONB,
    subtitle_srt_url VARCHAR(1000),
    subtitle_vtt_url VARCHAR(1000),
    subtitle_ass_url VARCHAR(1000),
    video_with_subs_url VARCHAR(1000),
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔑 API Спецификация

### Аутентификация

**POST /api/auth/register**
```json
Request:
{
  "email": "user@example.com",
  "password": "secure_password",
  "full_name": "John Doe"
}

Response:
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer"
}
```

**POST /api/auth/login**
```json
Request:
{
  "email": "user@example.com",
  "password": "secure_password"
}

Response:
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer"
}
```

### Работа с файлами

**POST /api/upload**
```
Headers:
  Authorization: Bearer {token}
  Content-Type: multipart/form-data

Body:
  file: <binary>

Response:
{
  "file_id": "uuid",
  "filename": "video.mp4",
  "size": 45234567,
  "mime_type": "video/mp4",
  "status": "uploaded"
}
```

**POST /api/transcribe**
```json
Headers:
  Authorization: Bearer {token}

Request:
{
  "file_id": "uuid",
  "quality": "standard",
  "target_lang": "ru"
}

Response:
{
  "task_id": "uuid",
  "status": "processing",
  "estimated_time": 90
}
```

**GET /api/status/{task_id}**
```json
Response:
{
  "task_id": "uuid",
  "status": "processing",
  "progress": 45,
  "current_stage": "translation",
  "eta": 60
}
```

**GET /api/result/{task_id}**
```json
Response:
{
  "task_id": "uuid",
  "original_text": "Сәлеметсіз бе...",
  "translated_text": "Здравствуйте...",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Сәлеметсіз бе",
      "translation": "Здравствуйте"
    }
  ],
  "subtitles": {
    "srt": "http://.../subtitles.srt",
    "vtt": "http://.../subtitles.vtt",
    "ass": "http://.../subtitles.ass"
  }
}
```

### WebSocket

**WS /ws/progress/{task_id}**
```json
Messages (server → client):
{
  "type": "progress",
  "progress": 45,
  "current_stage": "translation",
  "eta": 60,
  "status": "processing"
}

{
  "type": "completed",
  "task_id": "uuid"
}

{
  "type": "error",
  "error": "Processing failed"
}
```

---

## 🤖 ML Pipeline Модули

### 1. Audio Extractor

```python
from audio_extractor import AudioExtractor

extractor = AudioExtractor(sample_rate=16000)

# Извлечение аудио из видео
audio_path = extractor.extract_audio("video.mp4")

# Удаление шума
clean_audio = extractor.reduce_noise(audio_path)

# Нормализация громкости
normalized_audio = extractor.normalize_volume(clean_audio)

# Полный pipeline
processed_audio = extractor.process("video.mp4", clean_audio=True)
```

### 2. Whisper ASR

```python
from whisper_asr import WhisperASR

asr = WhisperASR(model_size="large-v3", device="cuda")

result = asr.transcribe("audio.wav", language="kk")

# result = {
#     "language": "kk",
#     "language_probability": 0.98,
#     "duration": 120.5,
#     "full_text": "Сәлеметсіз бе...",
#     "segments": [...]
# }
```

### 3. Text Normalizer

```python
from text_normalizer import TextNormalizer

normalizer = TextNormalizer(model="gpt-4o")

# С контекстом предыдущих сегментов
normalized = normalizer.normalize(
    text="қала га барамын",
    context=["Мен Астанада тұрамын"]
)
# → "қалаға барамын"
```

### 4. Multi-Model Translator

```python
from translator import MultiModelTranslator

translator = MultiModelTranslator()

result = translator.translate(
    text="Сәлеметсіз бе!",
    quality="premium",  # fast/standard/premium
    context=["Менің атым Айдар"]
)

# result = {
#     "translation": "Здравствуйте!",
#     "model": "gpt4o",
#     "confidence": 0.95
# }
```

### 5. Subtitle Generator

```python
from subtitle_generator import SubtitleGenerator

sub_gen = SubtitleGenerator()

# SRT формат
srt = sub_gen.generate_srt(segments, translations)

# WebVTT формат
vtt = sub_gen.generate_vtt(segments, translations)

# ASS формат (двуязычный)
ass = sub_gen.generate_ass(segments, translations)
```

---

## ⚙️ Конфигурация

### Backend (.env)

```env
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-256-bit-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Storage
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=media-files

# API Keys
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# ML
ML_WORKER_URL=http://localhost:5000
WHISPER_MODEL=large-v3  # tiny/base/small/medium/large-v3
```

### Frontend (.env)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## 🔧 Оптимизация производительности

### Whisper ASR

```python
# CPU оптимизация
asr = WhisperModel("large-v3", device="cpu", compute_type="int8")

# GPU оптимизация
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Меньшая модель для скорости
asr = WhisperModel("medium", device="cuda")

# Настройка beam search
result = asr.transcribe(
    audio,
    beam_size=3,  # 5 = точнее, 3 = быстрее
    vad_filter=True
)
```

### Кэширование

```python
# Redis кэш для частых запросов
@cache(ttl=3600)
async def get_translation(text: str, lang: str):
    return await translator.translate(text, lang)
```

### Параллельная обработка

```python
# Обработка нескольких сегментов параллельно
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    translations = list(executor.map(
        lambda seg: translator.translate(seg['text']),
        segments
    ))
```

---

## 📊 Мониторинг

### Метрики

```python
# Prometheus метрики
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
processing_time = Histogram('processing_seconds', 'Processing time')

@processing_time.time()
def process_video(video_path):
    request_count.inc()
    # ...
```

### Логирование

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Processing video", extra={"video_id": video_id})
```

---

## 🔒 Безопасность

### JWT Токены

```python
# Генерация токена
from jose import jwt
from datetime import datetime, timedelta

def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=30)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_file(...):
    ...
```

### Input Validation

```python
from pydantic import BaseModel, EmailStr, validator

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password too short')
        return v
```

---

## 🧪 Тестирование

### Unit Tests

```python
# tests/test_translator.py
import pytest
from translator import MultiModelTranslator

def test_translation():
    translator = MultiModelTranslator()
    result = translator.translate("Сәлем", quality="fast")
    assert result['translation'] == "Привет"
    assert result['confidence'] > 0.8
```

### Integration Tests

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_file():
    response = client.post(
        "/api/upload",
        files={"file": ("test.mp4", b"fake video data")},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "file_id" in response.json()
```

---

## 📈 Масштабирование

### Horizontal Scaling

```yaml
# docker-compose.yml для нескольких workers
services:
  worker:
    image: backend:latest
    scale: 4  # 4 worker экземпляра
    depends_on:
      - redis
```

### Load Balancing

```nginx
# nginx.conf
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

---

## 🔍 Отладка

### Debug Mode

```python
# app/main.py
app = FastAPI(debug=True)

# Включает:
# - Детальные ошибки
# - Auto-reload
# - Интерактивные трейсбеки
```

### Профилирование

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Ваш код

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

---

## 🎓 Best Practices

1. **Используйте виртуальные окружения**
2. **Пишите тесты** для критичного кода
3. **Логируйте** все важные события
4. **Валидируйте** входные данные
5. **Кэшируйте** частые запросы
6. **Мониторьте** производительность
7. **Документируйте** изменения
8. **Делайте резервные копии** БД
9. **Обновляйте** зависимости регулярно
10. **Тестируйте** перед деплоем

---

## 📚 Дополнительные ресурсы

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Next.js Docs**: https://nextjs.org/docs
- **Whisper Paper**: https://arxiv.org/abs/2212.04356
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **Docker Docs**: https://docs.docker.com/

---

Документация создана: 2026-02-25
Версия проекта: 1.0.0
