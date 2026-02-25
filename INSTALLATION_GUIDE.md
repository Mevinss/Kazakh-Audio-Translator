# 📦 Полная инструкция по развертыванию проекта

## ✅ Содержимое архива

Проект состоит из следующих компонентов:

```
kazakh-media-translator/
├── backend/              # FastAPI сервер
├── frontend/             # Next.js фронтенд  
├── ml_pipeline/          # ML модули и Jupyter notebooks
├── docker/               # Docker конфигурации
├── workers/              # Celery workers (фоновая обработка)
├── docker-compose.yml    # Оркестрация всех сервисов
├── README.md             # Общая документация
└── QUICKSTART.md         # Быстрый старт
```

---

## 🚀 СПОСОБ 1: Быстрый запуск с Docker (5 минут)

### Требования
- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM минимум
- 20GB свободного места

### Шаги

1. **Распакуйте архив**
```bash
unzip kazakh-media-translator.zip
cd kazakh-media-translator
```

2. **Настройте переменные окружения**
```bash
# Backend
cp backend/.env.example backend/.env

# Откройте backend/.env и добавьте ваши API ключи:
nano backend/.env

# Добавьте:
OPENAI_API_KEY=sk-proj-ваш-ключ
ANTHROPIC_API_KEY=sk-ant-api03-ваш-ключ
GOOGLE_API_KEY=AIzaSy-ваш-ключ
```

Где получить ключи:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys
- **Google AI**: https://aistudio.google.com/app/apikey

3. **Запустите все сервисы**
```bash
docker-compose up -d
```

4. **Проверьте статус**
```bash
docker-compose ps

# Должны быть running:
# - postgres
# - redis
# - minio
# - backend
# - frontend
# - worker
```

5. **Откройте в браузере**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

6. **Проверьте здоровье**
```bash
curl http://localhost:8000/health
```

### Готово! ✅

Перейдите на http://localhost:3000 и попробуйте загрузить тестовое видео.

---

## 🔧 СПОСОБ 2: Локальная разработка (без Docker)

Используйте этот способ если хотите модифицировать код.

### Требования
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- FFmpeg

### Backend Setup

```bash
cd backend

# Виртуальное окружение
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt

# Настройка .env
cp .env.example .env
# Отредактируйте .env (добавьте API ключи)

# Запуск PostgreSQL и Redis (если нет локально)
docker run -d -p 5432:5432 \
  -e POSTGRES_DB=kazakh_translator \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  postgres:15-alpine

docker run -d -p 6379:6379 redis:7-alpine

# Запуск сервера
uvicorn app.main:app --reload --port 8000
```

Backend доступен: http://localhost:8000

### Frontend Setup

```bash
cd frontend

# Установка зависимостей
npm install

# Настройка .env
cp .env.example .env

# Запуск dev сервера
npm run dev
```

Frontend доступен: http://localhost:3000

### ML Pipeline Setup

```bash
cd ml_pipeline

# Установка зависимостей
pip install -r requirements.txt

# Установка FFmpeg (если нет)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: https://ffmpeg.org/download.html

# Запуск Jupyter Lab
jupyter lab
```

Jupyter Lab: http://localhost:8888

---

## 🧪 Тестирование установки

### 1. Тест Backend API

```bash
# Health check
curl http://localhost:8000/health

# Ожидается:
# {"status":"healthy","redis":"connected","version":"1.0.0"}

# Регистрация пользователя
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123",
    "full_name": "Test User"
  }'

# Ожидается:
# {"access_token":"eyJ...","token_type":"bearer"}
```

### 2. Тест ML Pipeline

Создайте файл `test_ml.py`:

```python
import sys
sys.path.append('ml_pipeline/modules')

from audio_extractor import AudioExtractor
from whisper_asr import WhisperASR

# Тест извлечения аудио
print("Тест 1: Audio Extractor")
extractor = AudioExtractor()
print("✅ AudioExtractor инициализирован")

# Тест Whisper (требует GPU или займет время на CPU)
print("\nТест 2: Whisper ASR")
asr = WhisperASR(model_size="tiny")  # Используем tiny для быстрого теста
print("✅ WhisperASR инициализирован")

print("\n🎉 Все тесты пройдены!")
```

Запустите:
```bash
python test_ml.py
```

### 3. Тест с реальным видео

```bash
# Скачайте тестовое видео (или используйте свое)
cd ml_pipeline
python example_usage.py /path/to/your/video.mp4 standard
```

---

## 🔑 Получение API ключей

### OpenAI (GPT-4o)

1. Перейдите: https://platform.openai.com/
2. Sign Up или Log In
3. Settings → API keys
4. Create new secret key
5. Скопируйте ключ (показывается один раз!)
6. Формат: `sk-proj-...`
7. Пополните баланс минимум $5

### Anthropic (Claude 3.5 Sonnet)

1. Перейдите: https://console.anthropic.com/
2. Sign Up (нужна рабочая почта)
3. Settings → API Keys
4. Create Key
5. Формат: `sk-ant-api03-...`
6. Бесплатные $5 кредитов при регистрации

### Google AI (Gemini 1.5 Flash)

1. Перейдите: https://aistudio.google.com/
2. Войдите через Google аккаунт
3. Get API key → Create API key
4. Формат: `AIzaSy...`
5. **БЕСПЛАТНО**: 15 запросов/минуту

---

## 📊 Мониторинг и логи

### Docker логи

```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f worker

# Последние 100 строк
docker-compose logs --tail=100 backend
```

### Проверка ресурсов

```bash
# Использование ресурсов
docker stats

# Место на диске
docker system df
```

---

## 🔄 Обновление и перезапуск

### Docker

```bash
# Остановка
docker-compose down

# Обновление образов
docker-compose pull

# Пересборка (если изменили код)
docker-compose build

# Запуск
docker-compose up -d
```

### Локальная разработка

```bash
# Backend
cd backend
git pull
pip install -r requirements.txt
# Перезапустите uvicorn

# Frontend
cd frontend  
git pull
npm install
# Перезапустите npm run dev
```

---

## 🐛 Решение проблем

### Проблема: Backend не запускается

```bash
# Проверьте логи
docker-compose logs backend

# Типичные причины:
# 1. PostgreSQL не готов
docker-compose ps postgres

# 2. Неверные переменные окружения
cat backend/.env

# 3. Порт 8000 занят
lsof -i :8000
```

### Проблема: Frontend ошибка "Cannot connect to API"

```bash
# Проверьте backend
curl http://localhost:8000/health

# Проверьте .env
cat frontend/.env
# Должно быть: NEXT_PUBLIC_API_URL=http://localhost:8000

# Перезапустите frontend
docker-compose restart frontend
```

### Проблема: "Out of memory" при Whisper

```python
# В ml_pipeline/modules/whisper_asr.py
# Измените на меньшую модель:
asr = WhisperASR(model_size="medium")  # вместо large-v3

# Или используйте CPU:
asr = WhisperASR(model_size="large-v3", device="cpu")
```

### Проблема: Медленная обработка

**Решения:**
1. Используйте GPU с CUDA
2. Уменьшите модель Whisper до "medium" или "small"
3. Используйте качество перевода "fast" вместо "premium"
4. Увеличьте RAM (минимум 16GB для large-v3)

---

## 📚 Дополнительные ресурсы

- **API документация**: http://localhost:8000/docs
- **Jupyter notebooks**: `ml_pipeline/notebooks/`
- **Примеры использования**: `ml_pipeline/example_usage.py`
- **README**: см. README.md в корне проекта

---

## 🆘 Поддержка

Если возникли проблемы:

1. Проверьте логи: `docker-compose logs`
2. Изучите документацию в README.md
3. Откройте issue на GitHub
4. Email: support@kazakh-translator.com
5. Telegram: @kazakh_translator

---

## ✅ Чеклист готовности

Перед использованием убедитесь:

- [ ] Все сервисы запущены (`docker-compose ps`)
- [ ] Health check проходит (`curl localhost:8000/health`)
- [ ] API ключи добавлены в `backend/.env`
- [ ] Frontend открывается (http://localhost:3000)
- [ ] Можете зарегистрироваться и войти
- [ ] Тестовая загрузка файла работает

---

## 🎉 Готово к использованию!

Теперь вы можете:
1. Загрузить казахское видео на http://localhost:3000
2. Получить автоматический перевод и субтитры
3. Скачать результаты в формате SRT, VTT, ASS

Удачи! 🚀
