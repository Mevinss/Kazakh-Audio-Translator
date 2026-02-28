import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
DATABASE_PATH = os.path.join(BASE_DIR, 'database.db')

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a', 'mp4', 'avi', 'mkv', 'mov', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

SECRET_KEY = os.environ.get('SECRET_KEY', 'kazakh-asr-dev-key-2024')

# Whisper language code for Kazakh
LANGUAGE = 'kk'

# File cleanup: days after which uploaded files are deleted
FILE_MAX_AGE_DAYS = 7

# Transcriber model names - 3 high-accuracy models for Kazakh ASR comparison
MODEL_WHISPER_MEDIUM = 'whisper_medium'
MODEL_FASTER_WHISPER = 'faster_whisper'
MODEL_WHISPER_LARGE_V3 = 'whisper_large_v3'

MODEL_DISPLAY_NAMES = {
    MODEL_WHISPER_MEDIUM: 'Whisper Medium',
    MODEL_FASTER_WHISPER: 'Faster-Whisper Large-v3',
    MODEL_WHISPER_LARGE_V3: 'Whisper Large-v3',
}

# Source language is always Kazakh for subtitle comparison
SOURCE_LANGUAGES = {
    'kk': 'Қазақ тілі',
}
