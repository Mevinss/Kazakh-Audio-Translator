"""
Kazakh ASR Comparison Web Application
======================================
A simple Flask app for transcribing Kazakh audio/video with three ASR models
and comparing their accuracy metrics (WER, CER).

Machine translation (English/Russian → Kazakh) is integrated as an optional
post-transcription step using Helsinki-NLP MarianMT models.

Run:
    python app.py
    Open http://localhost:5000
"""

import csv
import io
import logging
import os
import uuid
from datetime import datetime, timedelta

from flask import (
    Flask, flash, jsonify, redirect, render_template,
    request, send_file, url_for,
)
from werkzeug.utils import secure_filename

import config
from modules import database
from modules.asr_engine import ASREngine, MODEL_DISPLAY_NAMES as ASR_MODEL_DISPLAY_NAMES
from modules.audio_processor import AudioProcessor
from modules.metrics import calculate_wer, calculate_cer, calculate_bleu
from modules.transcribers.whisper_base import WhisperBaseTranscriber
from modules.transcribers.whisper_medium import WhisperMediumTranscriber
from modules.transcribers.faster_whisper import FasterWhisperTranscriber

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Ensure required directories exist
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.MODELS_FOLDER, exist_ok=True)

# Initialise database
database.init_db()

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_transcribers = {}
_asr_engine = ASREngine()
MAX_SUBTITLE_FILENAME_LENGTH = 60


def _get_transcriber(model_key: str):
    if model_key not in _transcribers:
        if model_key == config.MODEL_WHISPER_BASE:
            _transcribers[model_key] = WhisperBaseTranscriber()
        elif model_key == config.MODEL_WHISPER_MEDIUM:
            _transcribers[model_key] = WhisperMediumTranscriber()
        elif model_key == config.MODEL_FASTER_WHISPER:
            _transcribers[model_key] = FasterWhisperTranscriber()
        else:
            raise ValueError(f"Unknown model: {model_key}")
    return _transcribers[model_key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def _cleanup_old_files():
    """Remove uploaded files and DB records older than FILE_MAX_AGE_DAYS."""
    cutoff = datetime.now() - timedelta(days=config.FILE_MAX_AGE_DAYS)
    for fname in os.listdir(config.UPLOAD_FOLDER):
        fpath = os.path.join(config.UPLOAD_FOLDER, fname)
        if os.path.isfile(fpath):
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
                logger.info("Removed old upload: %s", fpath)
    database.delete_old_files(config.FILE_MAX_AGE_DAYS)


def _create_srt_file(filename: str, model_key: str, segments: list) -> str:
    if not segments:
        return ''

    def _fmt(sec: float) -> str:
        sec = float(sec)
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        seconds = int(sec % 60)
        millis = int((sec - int(sec)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    safe_base = secure_filename(filename)[:MAX_SUBTITLE_FILENAME_LENGTH]
    srt_name = f"{uuid.uuid4().hex}_{model_key}_{safe_base}.srt"
    srt_path = os.path.join(config.UPLOAD_FOLDER, srt_name)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{_fmt(seg.get('start', 0.0))} --> {_fmt(seg.get('end', 0.0))}\n")
            f.write(f"{seg.get('text', '').strip()}\n\n")
    return srt_name


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html',
                           models=ASR_MODEL_DISPLAY_NAMES,
                           source_languages=config.SOURCE_LANGUAGES)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Upload file and run selected models."""
    if 'file' not in request.files:
        flash('Файл таңдалмады.', 'danger')
        return redirect(url_for('index'))

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        flash('Файл таңдалмады.', 'danger')
        return redirect(url_for('index'))

    if not _allowed_file(uploaded_file.filename):
        flash('Файл түріне рұқсат жоқ.', 'danger')
        return redirect(url_for('index'))

    selected_models = request.form.getlist('models')
    if not selected_models:
        flash('Кемінде бір модель таңдаңыз.', 'warning')
        return redirect(url_for('index'))

    reference_text = request.form.get('reference', '').strip() or None
    source_language = request.form.get('source_language', 'kk').strip() or 'kk'
    translate_to_kk = bool(request.form.get('translate_to_kk'))
    apply_denoise = bool(request.form.get('apply_denoise'))
    apply_normalization = bool(request.form.get('apply_normalization'))

    # Save file
    # secure_filename strips non-ASCII (e.g. Cyrillic) characters.  When the
    # base name is entirely non-ASCII the extension is also lost, so a file
    # like "Видео.mp4" becomes just "mp4" instead of "video.mp4".  Preserve
    # the original extension explicitly so FFmpeg can detect the format and
    # prepare_audio() can identify video files correctly.
    original_ext = os.path.splitext(uploaded_file.filename)[1].lower()
    filename = secure_filename(uploaded_file.filename)
    # secure_filename strips all non-ASCII characters. For a Cyrillic-only name
    # like "казахский.wav" it returns just "wav" (the extension chars), so
    # os.path.splitext("wav") yields ("wav", "") with no extension.  Restore
    # the original extension in that case so FFmpeg can identify the format.
    if (original_ext
            and original_ext.lstrip('.') in config.ALLOWED_EXTENSIONS
            and not os.path.splitext(filename)[1]):
        filename = filename + original_ext
    # If secure_filename returns an empty string (fully non-ASCII name, no
    # extension at all), give the file a safe generic name.
    if not filename or not filename.strip('._'):
        filename = 'upload' + original_ext
    unique_prefix = uuid.uuid4().hex
    filename = f"{unique_prefix}_{filename}"
    file_path = os.path.join(config.UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)
    logger.info("Saved upload: %s", file_path)

    # Prepare audio
    processor = AudioProcessor()
    try:
        audio_path = processor.prepare_audio(file_path, denoise=apply_denoise)
    except RuntimeError as exc:
        # FFmpeg not available or extraction failed – try using the raw file
        logger.warning("Audio preparation failed (%s); using raw file.", exc)
        audio_path = file_path

    # Transcribe with each selected model
    results = []
    for model_key in selected_models:
        if model_key not in config.MODEL_DISPLAY_NAMES and model_key not in ASR_MODEL_DISPLAY_NAMES:
            continue
        try:
            if model_key in ASR_MODEL_DISPLAY_NAMES:
                result = _asr_engine.process(
                    model_key=model_key,
                    audio_path=audio_path,
                    language=source_language,
                    apply_normalization=apply_normalization,
                )
            else:
                transcriber = _get_transcriber(model_key)
                result = transcriber.transcribe(audio_path, language=source_language)
                result['text_raw'] = result.get('text', '')
                detected_lang = result.get('language', source_language) or source_language
                translation_text = result.get('text', '') if detected_lang == 'kk' else ''
                if translate_to_kk and detected_lang in config.TRANSLATABLE_LANGUAGES:
                    try:
                        from modules.translator import get_translator
                        translation_text = get_translator().translate(
                            result.get('text', ''), src=detected_lang
                        )
                    except Exception as exc:
                        logger.warning("Legacy translation failed: %s", exc)
                result['translation'] = translation_text
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", model_key, exc)
            result = {
                'text': f'Error: {exc}',
                'text_raw': '',
                'translation': '',
                'segments': [],
                'duration': 0,
                'confidence': 0,
                'processing_time': 0,
                'language': source_language,
            }

        wer_val = cer_val = bleu_val = None
        if reference_text and result['text']:
            try:
                wer_val = calculate_wer(reference_text, result['text'])
                cer_val = calculate_cer(reference_text, result['text'])
            except Exception as exc:
                logger.warning("Metrics calculation failed: %s", exc)

        detected_lang = result.get('language', source_language) or source_language
        translation_text = result.get('translation') or ''
        if reference_text:
            bleu_source = translation_text or (result['text'] if detected_lang == 'kk' else '')
            if bleu_source:
                bleu_val = calculate_bleu(reference_text, bleu_source)
        subtitle_file = _create_srt_file(uploaded_file.filename, model_key, result.get('segments', []))

        record_id = database.save_transcription(
            filename=filename,
            model=model_key,
            text=result['text'],
            duration=result['duration'],
            confidence=result['confidence'],
            wer=wer_val,
            cer=cer_val,
            bleu=bleu_val,
            reference=reference_text,
            translation=translation_text,
            source_language=detected_lang,
        )
        results.append({
            'id': record_id,
            'model_key': model_key,
            'model_name': config.MODEL_DISPLAY_NAMES.get(
                model_key, ASR_MODEL_DISPLAY_NAMES.get(model_key, model_key)
            ),
            'text': result['text'],
            'text_raw': result.get('text_raw'),
            'segments': result['segments'],
            'duration': result['duration'],
            'confidence': result['confidence'],
            'processing_time': result['processing_time'],
            'wer': wer_val,
            'cer': cer_val,
            'bleu': bleu_val,
            'translation': translation_text,
            'subtitle_file': subtitle_file,
            'detected_language': detected_lang,
        })

    # Warn when every model produced no text (common first-run symptom)
    if results and all(not r['text'] or r['text'].startswith('Error:') for r in results):
        flash(
            'Барлық модель бос нәтиже қайтарды. '
            'Аудиода сөйлеу бар екенін және FFmpeg орнатылғанын тексеріңіз. '
            'Алғашқы іске қосуда модельдер жүктелуі мүмкін, бұл бірнеше минут алады.',
            'warning',
        )

    _cleanup_old_files()

    return render_template(
        'results.html',
        filename=uploaded_file.filename,
        results=results,
        reference=reference_text,
        source_language=source_language,
        language_names=config.SOURCE_LANGUAGES,
    )


@app.route('/results/<int:record_id>')
def result_detail(record_id: int):
    record = database.get_transcription(record_id)
    if record is None:
        flash('Жазба табылмады.', 'danger')
        return redirect(url_for('history'))
    return render_template('results.html',
                           filename=record['filename'],
                           results=[{
                                'id': record['id'],
                                'model_key': record['model'],
                                'model_name': config.MODEL_DISPLAY_NAMES.get(
                                    record['model'],
                                    ASR_MODEL_DISPLAY_NAMES.get(record['model'], record['model']),
                                ),
                                'text': record['text'],
                                'text_raw': None,
                                'segments': [],
                                'duration': record['duration'],
                                'confidence': record['confidence'],
                                'processing_time': None,
                                'wer': record['wer'],
                                'cer': record['cer'],
                                'bleu': record.get('bleu'),
                                'translation': record.get('translation'),
                                'subtitle_file': '',
                                'detected_language': record.get('source_language'),
                            }],
                           reference=record.get('reference'),
                           source_language=record.get('source_language', 'kk'),
                           language_names=config.SOURCE_LANGUAGES)


@app.route('/history')
def history():
    records = database.get_all_transcriptions()
    return render_template('history.html', records=records,
                           model_names=config.MODEL_DISPLAY_NAMES)


@app.route('/export')
def export_csv():
    """Export all transcription history as CSV."""
    records = database.get_all_transcriptions()

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=['id', 'filename', 'model', 'text', 'wer', 'cer', 'bleu',
                    'duration', 'confidence', 'created_at'],
        extrasaction='ignore',
    )
    writer.writeheader()
    writer.writerows(records)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='transcriptions.csv',
    )


@app.route('/export/<int:record_id>')
def export_single_csv(record_id: int):
    """Export a single transcription record as CSV."""
    record = database.get_transcription(record_id)
    if record is None:
        flash('Жазба табылмады.', 'danger')
        return redirect(url_for('history'))

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=['id', 'filename', 'model', 'text', 'wer', 'cer', 'bleu',
                    'duration', 'confidence', 'created_at'],
        extrasaction='ignore',
    )
    writer.writeheader()
    writer.writerow(record)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'transcription_{record_id}.csv',
    )


@app.route('/subtitles/<path:srt_name>')
def download_subtitles(srt_name: str):
    safe_name = os.path.basename(srt_name)
    srt_path = os.path.join(config.UPLOAD_FOLDER, safe_name)
    if not os.path.isfile(srt_path):
        flash('Субтитр файлы табылмады.', 'danger')
        return redirect(url_for('history'))
    return send_file(
        srt_path,
        mimetype='application/x-subrip',
        as_attachment=True,
        download_name=safe_name,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=5000)
