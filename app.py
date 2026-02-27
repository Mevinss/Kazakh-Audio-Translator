"""
Kazakh ASR Comparison Web Application
======================================
A simple Flask app for transcribing Kazakh audio/video with three ASR models
and comparing their accuracy metrics (WER, CER).

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
from modules.audio_processor import AudioProcessor
from modules.metrics import calculate_wer, calculate_cer
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html',
                           models=config.MODEL_DISPLAY_NAMES)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Upload file and run selected models."""
    if 'file' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('index'))

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('index'))

    if not _allowed_file(uploaded_file.filename):
        flash('File type not allowed.', 'danger')
        return redirect(url_for('index'))

    selected_models = request.form.getlist('models')
    if not selected_models:
        flash('Please select at least one model.', 'warning')
        return redirect(url_for('index'))

    reference_text = request.form.get('reference', '').strip() or None

    # Save file
    filename = secure_filename(uploaded_file.filename)
    unique_prefix = uuid.uuid4().hex
    filename = f"{unique_prefix}_{filename}"
    file_path = os.path.join(config.UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)
    logger.info("Saved upload: %s", file_path)

    # Prepare audio
    processor = AudioProcessor()
    try:
        audio_path = processor.prepare_audio(file_path)
    except RuntimeError as exc:
        # FFmpeg not available or extraction failed – try using the raw file
        logger.warning("Audio preparation failed (%s); using raw file.", exc)
        audio_path = file_path

    # Transcribe with each selected model
    results = []
    for model_key in selected_models:
        if model_key not in config.MODEL_DISPLAY_NAMES:
            continue
        try:
            transcriber = _get_transcriber(model_key)
            result = transcriber.transcribe(audio_path)
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", model_key, exc)
            result = {
                'text': f'Error: {exc}',
                'segments': [],
                'duration': 0,
                'confidence': 0,
                'processing_time': 0,
            }

        wer_val = cer_val = None
        if reference_text and result['text']:
            try:
                wer_val = calculate_wer(reference_text, result['text'])
                cer_val = calculate_cer(reference_text, result['text'])
            except Exception as exc:
                logger.warning("Metrics calculation failed: %s", exc)

        record_id = database.save_transcription(
            filename=filename,
            model=model_key,
            text=result['text'],
            duration=result['duration'],
            confidence=result['confidence'],
            wer=wer_val,
            cer=cer_val,
            reference=reference_text,
        )
        results.append({
            'id': record_id,
            'model_key': model_key,
            'model_name': config.MODEL_DISPLAY_NAMES[model_key],
            'text': result['text'],
            'segments': result['segments'],
            'duration': result['duration'],
            'confidence': result['confidence'],
            'processing_time': result['processing_time'],
            'wer': wer_val,
            'cer': cer_val,
        })

    _cleanup_old_files()

    return render_template(
        'results.html',
        filename=uploaded_file.filename,
        results=results,
        reference=reference_text,
    )


@app.route('/results/<int:record_id>')
def result_detail(record_id: int):
    record = database.get_transcription(record_id)
    if record is None:
        flash('Record not found.', 'danger')
        return redirect(url_for('history'))
    return render_template('results.html',
                           filename=record['filename'],
                           results=[{
                               'id': record['id'],
                               'model_key': record['model'],
                               'model_name': config.MODEL_DISPLAY_NAMES.get(record['model'], record['model']),
                               'text': record['text'],
                               'segments': [],
                               'duration': record['duration'],
                               'confidence': record['confidence'],
                               'processing_time': None,
                               'wer': record['wer'],
                               'cer': record['cer'],
                           }],
                           reference=record.get('reference'))


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
        fieldnames=['id', 'filename', 'model', 'text', 'wer', 'cer',
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
        flash('Record not found.', 'danger')
        return redirect(url_for('history'))

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=['id', 'filename', 'model', 'text', 'wer', 'cer',
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=5000)
