# 📦 Инструкция по установке и запуску

> ⚡ **Кратко:** Docker для этого проекта **не нужен**.
> Приложение — это обычный веб-сервер на Python (Flask), который запускается
> командой `python app.py`. Для работы достаточно Python 3.9+, FFmpeg и
> нескольких pip-пакетов.

---

## ✅ Структура проекта

```
Kazakh-Audio-Translator/
├── app.py                       # Flask-приложение (точка входа)
├── config.py                    # Конфигурация
├── requirements.txt             # Python-зависимости
├── modules/
│   ├── audio_processor.py       # Извлечение / нормализация аудио
│   ├── metrics.py               # WER / CER
│   ├── database.py              # SQLite (история транскрибирований)
│   └── transcribers/
│       ├── whisper_base.py      # Whisper Base (74 M параметров)
│       ├── whisper_medium.py    # Whisper Medium (307 M)
│       └── faster_whisper.py    # Faster-Whisper Large-v3
├── templates/                   # HTML-шаблоны
├── static/                      # CSS / JS
├── uploads/                     # Загружаемые файлы (авто-очистка)
└── models/                      # Кэш скачанных моделей
```

> **Примечание:** папки `backend/`, `frontend/`, `ml_pipeline/` и файл
> `docker-compose.yml` в этом репозитории **отсутствуют** — они не нужны.
> Всё приложение находится прямо в корне репозитория.

---

## 🚀 Быстрый старт (без Docker)

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/Mevinss/Kazakh-Audio-Translator.git
cd Kazakh-Audio-Translator
```

### 2. Создайте и активируйте виртуальное окружение

```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Установите Python-зависимости

```bash
pip install -r requirements.txt
```

> Установка может занять несколько минут — скачиваются PyTorch и Whisper.

### 4. Установите FFmpeg

FFmpeg нужен для извлечения аудио из видеофайлов (mp4, avi, mkv и т. д.).

**Windows**

1. Перейдите на https://ffmpeg.org/download.html → Windows builds.
2. Скачайте архив, распакуйте, например, в `C:\ffmpeg`.
3. Добавьте `C:\ffmpeg\bin` в переменную среды `PATH`:
   - Пуск → «Переменные среды» → `Path` → **Изменить** → **Создать** →
     вставьте путь → **ОК**.
4. Откройте новый терминал и проверьте: `ffmpeg -version`.

**Linux (Ubuntu / Debian)**

```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS**

```bash
brew install ffmpeg
```

### 5. Запустите приложение

```bash
python app.py
```

Откройте в браузере: **http://localhost:5000**

---

## 🖥️ Что будет при первом запуске

При первом открытии страницы и выборе модели Whisper автоматически
скачиваются веса из интернета и сохраняются в папке `models/`:

| Модель | Размер файла | RAM |
|--------|-------------|-----|
| Whisper Base | ~140 МБ | ~1 ГБ |
| Whisper Medium | ~1.4 ГБ | ~5 ГБ |
| Faster-Whisper Large-v3 | ~3 ГБ | ~10 ГБ |

При последующих запусках модели загружаются с диска мгновенно.

---

## 🧪 Проверка установки

```bash
# Должен открыться браузер или вернуть HTML:
curl http://localhost:5000

# Ожидается HTML главной страницы (не ошибка).
```

Или просто откройте http://localhost:5000 в браузере — должна появиться
страница загрузки файла.

---

## 🐳 Использование Docker (опционально)

Docker для этого проекта **не обязателен** — вы уже можете запустить
приложение командой `python app.py` (см. выше).

Если вы всё равно хотите использовать Docker, создайте `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Сборка и запуск:

```bash
docker build -t kazakh-asr .
docker run -p 5000:5000 kazakh-asr
```

---

## 📊 Мониторинг и логи

```bash
# Логи Flask выводятся прямо в терминал при запуске python app.py.
# Для режима отладки:
FLASK_DEBUG=1 python app.py      # Linux / macOS
set FLASK_DEBUG=1 && python app.py  # Windows (cmd)
$env:FLASK_DEBUG=1; python app.py   # Windows (PowerShell)
```

---

## 🔄 Обновление

```bash
git pull
pip install -r requirements.txt   # обновить зависимости, если изменились
python app.py
```

---

## 🐛 Решение проблем

### Проблема: «Нет папки backend / frontend / ml_pipeline»

Эти папки **не существуют в данном проекте**. Если вы встречали инструкцию,
которая предлагает перейти в `cd backend` или запустить `docker-compose` —
она описывала другой, более сложный проект. В этом репозитории всё приложение
находится прямо в корне: `app.py`, `config.py`, `requirements.txt`.

**Правильная команда запуска:**

```bash
# Из корня репозитория:
python app.py
```

---

### Проблема: «command not found: ffmpeg» или ошибка при обработке видео

Установите FFmpeg по инструкции из раздела «Быстрый старт → Шаг 4» выше.

---

### Проблема: «Out of memory» при Whisper

```python
# Откройте modules/transcribers/whisper_medium.py и уменьшите модель:
# Замените "medium" на "small" или "base"
```

Или запустите только модель Whisper Base (самую лёгкую) на странице
транскрибирования.

---

### Проблема: Медленная обработка

**Решения:**
1. Выберите модель **Whisper Base** вместо Medium или Faster-Whisper Large-v3.
2. При наличии NVIDIA-видеокарты убедитесь, что установлен CUDA-совместимый PyTorch:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. На CPU обработка 1 минуты аудио занимает ~2–10 минут в зависимости от модели.

---

### Проблема: Docker занимает слишком много места на диске C

Docker для этого проекта не нужен. Если он уже установлен и занимает много
места, ниже описано, как освободить место или перенести данные на другой диск.

#### Шаг 1 — Проверить использование

```bash
docker system df
```

#### Шаг 2 — Освободить место (очистка)

```bash
# Удалить остановленные контейнеры, неиспользуемые образы, сети и кэш:
docker system prune -a

# Удалить только неиспользуемые тома (ОСТОРОЖНО — данные БД!):
docker volume prune

# Удалить только кэш сборок:
docker builder prune -a
```

> ⚠️ `docker system prune -a` удаляет **все** образы, которые не используются
> запущенными контейнерами.

---

#### Перенос Docker на диск D (Windows)

Docker Desktop хранит данные внутри WSL 2 — файлов `ext4.vhdx` на диске C.
Ниже два способа перенести их на диск D.

##### Способ А — через настройки Docker Desktop (рекомендуется)

1. Откройте **Docker Desktop**.
2. Перейдите в **Settings → Resources → Advanced**.
3. В поле **Disk image location** нажмите **Browse** и выберите папку на диске D,
   например `D:\DockerData`.
4. Нажмите **Apply & Restart**.

##### Способ Б — перенос через WSL 2 вручную

> **Требования:** PowerShell (с правами администратора), WSL 2.

```powershell
# 1. Закройте Docker Desktop (трей → Quit Docker Desktop), остановите WSL:
wsl --shutdown

# 2. Убедитесь, что дистрибутивы видны:
wsl --list --verbose

# 3. Создайте папки на диске D:
New-Item -ItemType Directory -Path "D:\DockerWSL\data"
New-Item -ItemType Directory -Path "D:\DockerWSL\desktop"

# 4. Экспортируйте дистрибутивы:
wsl --export docker-desktop-data "D:\DockerWSL\docker-desktop-data.tar"
wsl --export docker-desktop      "D:\DockerWSL\docker-desktop.tar"

# 5. Удалите с диска C:
wsl --unregister docker-desktop-data
wsl --unregister docker-desktop

# 6. Импортируйте на диск D:
wsl --import docker-desktop-data "D:\DockerWSL\data"    "D:\DockerWSL\docker-desktop-data.tar" --version 2
wsl --import docker-desktop      "D:\DockerWSL\desktop" "D:\DockerWSL\docker-desktop.tar"      --version 2

# 7. Запустите Docker Desktop — данные теперь на диске D.

# 8. Удалите временные tar-файлы:
Remove-Item "D:\DockerWSL\docker-desktop-data.tar"
Remove-Item "D:\DockerWSL\docker-desktop.tar"
```

---

#### Ограничение размера виртуального диска WSL 2

Создайте (или отредактируйте) файл `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
diskSizeGB=60
```

Затем выполните `wsl --shutdown` и перезапустите Docker Desktop.

---

#### Перенос Docker на другой диск (Linux)

```bash
sudo systemctl stop docker
sudo rsync -aP /var/lib/docker/ /mnt/data/docker/

# В /etc/docker/daemon.json добавьте:
# { "data-root": "/mnt/data/docker" }

sudo systemctl start docker
docker info | grep "Docker Root Dir"
sudo rm -rf /var/lib/docker   # только после проверки
```

---

## 📚 Дополнительные ресурсы

- **README**: см. README.md в корне проекта
- **Whisper документация**: https://github.com/openai/whisper
- **Faster-Whisper**: https://github.com/SYSTRAN/faster-whisper
- **FFmpeg**: https://ffmpeg.org/documentation.html

---

## 🆘 Поддержка

Если возникли проблемы:

1. Убедитесь, что запускаете `python app.py` из корня репозитория.
2. Проверьте, что активировано виртуальное окружение (`venv`).
3. Откройте issue на GitHub.
4. Email: support@kazakh-translator.com
5. Telegram: @kazakh_translator

---

## ✅ Чеклист готовности

Перед использованием убедитесь:

- [ ] Python 3.9+ установлен (`python --version`)
- [ ] Виртуальное окружение активировано
- [ ] Зависимости установлены (`pip install -r requirements.txt`)
- [ ] FFmpeg установлен (`ffmpeg -version`)
- [ ] Приложение запущено (`python app.py`)
- [ ] Страница открывается (http://localhost:5000)

---

## 🎉 Готово к использованию!

Теперь вы можете:
1. Открыть http://localhost:5000
2. Загрузить казахский аудио- или видеофайл
3. Выбрать одну или несколько ASR-моделей для сравнения
4. Получить транскрипцию с метриками WER/CER
5. Экспортировать историю в CSV

Удачи! 🚀
