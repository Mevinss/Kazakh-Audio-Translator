"""
Tests for the Kazakh ASR comparison application.

These tests are designed to run without ML models installed by mocking
the heavy dependencies. They validate application logic, metrics,
database operations, and HTTP endpoints.
"""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Make sure the project root is on sys.path regardless of how tests are run
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def test_wer_identical(self):
        """WER should be 0 for identical strings."""
        from modules.metrics import calculate_wer
        self.assertEqual(calculate_wer("сәлем", "сәлем"), 0.0)

    def test_wer_completely_different(self):
        """WER should be 1 (or more) for completely different strings."""
        from modules.metrics import calculate_wer
        result = calculate_wer("сәлем", "hello world foo bar")
        self.assertGreater(result, 0.0)

    def test_cer_identical(self):
        """CER should be 0 for identical strings."""
        from modules.metrics import calculate_cer
        self.assertEqual(calculate_cer("тест", "тест"), 0.0)

    def test_cer_partial(self):
        """CER should be > 0 for partially different strings."""
        from modules.metrics import calculate_cer
        result = calculate_cer("сәлем", "сәлам")
        self.assertGreater(result, 0.0)

    def test_wer_empty_input(self):
        """WER should return None for empty strings."""
        from modules.metrics import calculate_wer
        self.assertIsNone(calculate_wer("", "some text"))
        self.assertIsNone(calculate_wer("some text", ""))
        self.assertIsNone(calculate_wer("", ""))

    def test_cer_empty_input(self):
        """CER should return None for empty strings."""
        from modules.metrics import calculate_cer
        self.assertIsNone(calculate_cer("", ""))


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase(unittest.TestCase):

    def setUp(self):
        """Create a fresh temporary SQLite database for each test."""
        self._tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp.close()
        # Patch the DATABASE_PATH used by the database module
        import config
        self._orig_path = config.DATABASE_PATH
        config.DATABASE_PATH = self._tmp.name

        import importlib
        import modules.database as db_module
        importlib.reload(db_module)
        self.db = db_module
        self.db.init_db()

    def tearDown(self):
        import config
        config.DATABASE_PATH = self._orig_path
        os.unlink(self._tmp.name)

    def test_save_and_retrieve(self):
        record_id = self.db.save_transcription(
            filename='test.wav',
            model='whisper_base',
            text='Сәлеметсіз бе',
            duration=5.0,
            confidence=0.85,
        )
        record = self.db.get_transcription(record_id)
        self.assertIsNotNone(record)
        self.assertEqual(record['filename'], 'test.wav')
        self.assertEqual(record['model'], 'whisper_base')
        self.assertEqual(record['text'], 'Сәлеметсіз бе')
        self.assertAlmostEqual(record['confidence'], 0.85)

    def test_save_with_metrics(self):
        record_id = self.db.save_transcription(
            filename='audio.mp3',
            model='faster_whisper',
            text='Жақсы',
            duration=3.0,
            confidence=0.9,
            wer=0.12,
            cer=0.05,
            reference='жақсы',
        )
        record = self.db.get_transcription(record_id)
        self.assertAlmostEqual(record['wer'], 0.12)
        self.assertAlmostEqual(record['cer'], 0.05)

    def test_get_all_transcriptions(self):
        for i in range(3):
            self.db.save_transcription(
                filename=f'file{i}.wav',
                model='whisper_base',
                text=f'text {i}',
                duration=1.0,
                confidence=0.7,
            )
        records = self.db.get_all_transcriptions()
        self.assertEqual(len(records), 3)

    def test_get_nonexistent(self):
        self.assertIsNone(self.db.get_transcription(99999))


# ---------------------------------------------------------------------------
# Flask app / route tests (no ML)
# ---------------------------------------------------------------------------

class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        """Set up Flask test client with a temporary database."""
        import config

        self._tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp_db.close()
        self._tmp_upload = tempfile.mkdtemp()

        self._orig_db = config.DATABASE_PATH
        self._orig_upload = config.UPLOAD_FOLDER

        config.DATABASE_PATH = self._tmp_db.name
        config.UPLOAD_FOLDER = self._tmp_upload

        # Reload database module so it uses the new path
        import importlib
        import modules.database as db_module
        importlib.reload(db_module)

        import app as flask_app
        importlib.reload(flask_app)

        flask_app.app.config['TESTING'] = True
        flask_app.app.config['WTF_CSRF_ENABLED'] = False
        self.client = flask_app.app.test_client()

    def tearDown(self):
        import config
        config.DATABASE_PATH = self._orig_db
        config.UPLOAD_FOLDER = self._orig_upload
        os.unlink(self._tmp_db.name)
        import shutil
        shutil.rmtree(self._tmp_upload, ignore_errors=True)

    def test_index_returns_200(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Kazakh ASR', response.data)

    def test_history_returns_200(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)

    def test_transcribe_no_file_redirects(self):
        response = self.client.post('/transcribe', data={})
        self.assertIn(response.status_code, (302, 200))

    def test_transcribe_with_file_and_mock_model(self):
        """Upload an audio file and mock the transcriber to avoid loading models."""
        fake_audio = b'RIFF' + b'\x00' * 100  # fake WAV bytes

        mock_result = {
            'text': 'Сәлеметсіз бе',
            'segments': [{'start': 0.0, 'end': 2.0, 'text': 'Сәлеметсіз бе'}],
            'duration': 2.0,
            'confidence': 0.9,
            'processing_time': 1.5,
            'language': 'kk',
        }

        import app as flask_app
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        with patch.object(flask_app, '_get_transcriber', return_value=mock_transcriber), \
             patch('modules.audio_processor.AudioProcessor.prepare_audio',
                   return_value='/tmp/fake_audio.wav'):
            response = self.client.post(
                '/transcribe',
                content_type='multipart/form-data',
                data={
                    'file': (io.BytesIO(fake_audio), 'test.wav'),
                    'models': ['whisper_base'],
                    'reference': '',
                },
            )
        # Should show results page or redirect
        self.assertIn(response.status_code, (200, 302))

    def test_export_csv(self):
        """Export endpoint should return a CSV file."""
        import modules.database as db_module
        db_module.save_transcription(
            filename='sample.wav',
            model='whisper_base',
            text='Тест',
            duration=3.0,
            confidence=0.8,
        )
        response = self.client.get('/export')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/csv', response.content_type)

    def test_result_detail_not_found(self):
        response = self.client.get('/results/99999')
        self.assertIn(response.status_code, (302, 200))


# ---------------------------------------------------------------------------
# BaseTranscriber interface tests
# ---------------------------------------------------------------------------

class TestBaseTranscriber(unittest.TestCase):

    def test_cannot_instantiate_abstract(self):
        from modules.transcribers.base_transcriber import BaseTranscriber
        with self.assertRaises(TypeError):
            BaseTranscriber()

    def test_concrete_subclass_works(self):
        from modules.transcribers.base_transcriber import BaseTranscriber

        class DummyTranscriber(BaseTranscriber):
            def transcribe(self, audio_path: str, language: str = 'kk') -> dict:
                return {'text': 'hello', 'segments': [], 'duration': 1.0,
                        'confidence': 1.0, 'language': language}

        t = DummyTranscriber()
        result = t.transcribe('fake.wav')
        self.assertEqual(result['text'], 'hello')
        self.assertEqual(result['language'], 'kk')

        result_en = t.transcribe('fake.wav', language='en')
        self.assertEqual(result_en['language'], 'en')


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):

    def test_model_keys_defined(self):
        import config
        self.assertIn(config.MODEL_WHISPER_BASE, config.MODEL_DISPLAY_NAMES)
        self.assertIn(config.MODEL_WHISPER_MEDIUM, config.MODEL_DISPLAY_NAMES)
        self.assertIn(config.MODEL_FASTER_WHISPER, config.MODEL_DISPLAY_NAMES)

    def test_allowed_extensions(self):
        import config
        self.assertIn('wav', config.ALLOWED_EXTENSIONS)
        self.assertIn('mp4', config.ALLOWED_EXTENSIONS)
        self.assertNotIn('txt', config.ALLOWED_EXTENSIONS)


# ---------------------------------------------------------------------------
# AudioProcessor tests (mocked FFmpeg)
# ---------------------------------------------------------------------------

class TestAudioProcessor(unittest.TestCase):
    """Tests for AudioProcessor that do not require a real FFmpeg binary."""

    def _make_processor(self):
        from modules.audio_processor import AudioProcessor
        return AudioProcessor()

    def test_normalize_audio_output_always_wav(self):
        """normalize_audio must produce a .wav output path regardless of input extension.

        Regression: previously the output extension was inherited from the
        input, so an extensionless input (e.g. '{uuid}_mp4', which happens
        when secure_filename strips a Cyrillic base name) produced an
        extensionless output and FFmpeg failed with
        "Unable to choose an output format".
        """
        proc = self._make_processor()

        # Patch subprocess.run so we don't need a real FFmpeg
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = None  # success

            # Case 1: extensionless input (Cyrillic filename stripped)
            out = proc.normalize_audio('/uploads/abc123_mp4')
            self.assertTrue(out.endswith('.wav'),
                            f"Expected .wav output, got: {out}")
            # The FFmpeg command must include the .wav output path
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[-1], out)
            self.assertIn('-acodec', cmd)
            self.assertIn('pcm_s16le', cmd)

            # Case 2: normal .wav input — still .wav
            out2 = proc.normalize_audio('/uploads/abc_extracted.wav')
            self.assertTrue(out2.endswith('.wav'),
                            f"Expected .wav output, got: {out2}")

            # Case 3: .mp3 input — now .wav
            out3 = proc.normalize_audio('/uploads/abc_audio.mp3')
            self.assertTrue(out3.endswith('.wav'),
                            f"Expected .wav output for mp3 input, got: {out3}")

    def test_prepare_audio_cyrillic_filename_calls_extract_for_video(self):
        """prepare_audio must call extract_audio for video files.

        Regression: when the file was saved with the extension restored after
        secure_filename stripped Cyrillic characters (e.g. 'abc_mp4.mp4'),
        prepare_audio must recognise it as a video and call extract_audio.
        """
        proc = self._make_processor()

        with patch.object(proc, 'extract_audio', return_value='/tmp/extracted.wav') as mock_extract, \
             patch.object(proc, 'normalize_audio', return_value='/tmp/extracted_normalized.wav') as mock_norm, \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=2048):
            result = proc.prepare_audio('/uploads/abc_mp4.mp4')
            mock_extract.assert_called_once_with('/uploads/abc_mp4.mp4')
            mock_norm.assert_called_once_with('/tmp/extracted.wav')
            self.assertEqual(result, '/tmp/extracted_normalized.wav')

    def test_prepare_audio_audio_only_skips_extract(self):
        """prepare_audio must NOT call extract_audio for audio-only files."""
        proc = self._make_processor()

        with patch.object(proc, 'extract_audio') as mock_extract, \
             patch.object(proc, 'normalize_audio', return_value='/tmp/n.wav'), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=1024):
            proc.prepare_audio('/uploads/abc.wav')
            mock_extract.assert_not_called()

    def test_prepare_audio_falls_back_when_normalized_is_empty(self):
        """prepare_audio must return the pre-normalized path when the normalized
        WAV is zero bytes or missing — prevents silent empty transcriptions."""
        proc = self._make_processor()

        with patch.object(proc, 'extract_audio') as mock_extract, \
             patch.object(proc, 'normalize_audio', return_value='/tmp/n.wav'), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=0):  # zero-byte output
            result = proc.prepare_audio('/uploads/abc.wav')
            # Should fall back to the original audio (input to normalize_audio)
            self.assertEqual(result, '/uploads/abc.wav')
            mock_extract.assert_not_called()

    def test_prepare_audio_falls_back_when_normalized_missing(self):
        """prepare_audio must return the pre-normalized path when the normalized
        file does not exist on disk.  Also verify getsize is not called when
        isfile already returns False (short-circuit)."""
        proc = self._make_processor()

        with patch.object(proc, 'normalize_audio', return_value='/tmp/n.wav'), \
             patch('os.path.isfile', return_value=False) as mock_isfile, \
             patch('os.path.getsize') as mock_getsize:
            result = proc.prepare_audio('/uploads/abc.wav')
            self.assertEqual(result, '/uploads/abc.wav')
            mock_isfile.assert_called_once_with('/tmp/n.wav')
            mock_getsize.assert_not_called()


class TestCyrillicFilenameHandling(unittest.TestCase):
    """Tests that verify the upload route preserves file extensions when
    secure_filename strips a Cyrillic base name."""

    def setUp(self):
        import config
        import importlib
        import tempfile

        self._tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp_db.close()
        self._tmp_upload = tempfile.mkdtemp()

        self._orig_db = config.DATABASE_PATH
        self._orig_upload = config.UPLOAD_FOLDER
        config.DATABASE_PATH = self._tmp_db.name
        config.UPLOAD_FOLDER = self._tmp_upload

        import modules.database as db_module
        importlib.reload(db_module)

        import app as flask_app
        importlib.reload(flask_app)

        flask_app.app.config['TESTING'] = True
        self.client = flask_app.app.test_client()
        self.flask_app = flask_app

    def tearDown(self):
        import config
        import shutil
        config.DATABASE_PATH = self._orig_db
        config.UPLOAD_FOLDER = self._orig_upload
        os.unlink(self._tmp_db.name)
        shutil.rmtree(self._tmp_upload, ignore_errors=True)

    def _upload(self, filename, content=b'fake'):
        import app as flask_app
        mock_result = {
            'text': 'test', 'segments': [], 'duration': 1.0,
            'confidence': 0.9, 'processing_time': 0.1,
            'language': 'kk',
        }
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        with patch.object(flask_app, '_get_transcriber', return_value=mock_transcriber), \
             patch('modules.audio_processor.AudioProcessor.prepare_audio',
                   return_value='/tmp/fake.wav'):
            self.client.post(
                '/transcribe',
                content_type='multipart/form-data',
                data={
                    'file': (io.BytesIO(content), filename),
                    'models': ['whisper_base'],
                },
            )

    def _saved_filenames(self):
        import config
        return os.listdir(config.UPLOAD_FOLDER)

    def test_cyrillic_mp4_preserves_extension(self):
        """File 'Видео.mp4' must be saved with a .mp4 extension."""
        self._upload('Видео.mp4')
        files = self._saved_filenames()
        self.assertEqual(len(files), 1, f"Expected 1 uploaded file, got: {files}")
        self.assertTrue(files[0].endswith('.mp4'),
                        f"Expected .mp4 extension, saved as: {files[0]}")

    def test_cyrillic_wav_preserves_extension(self):
        """File 'аудио.wav' must be saved with a .wav extension."""
        self._upload('аудио.wav')
        files = self._saved_filenames()
        self.assertEqual(len(files), 1, f"Expected 1 uploaded file, got: {files}")
        self.assertTrue(files[0].endswith('.wav'),
                        f"Expected .wav extension, saved as: {files[0]}")

    def test_ascii_filename_unchanged(self):
        """ASCII filename 'video.mp4' must continue to work correctly."""
        self._upload('video.mp4')
        files = self._saved_filenames()
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith('.mp4'),
                        f"Expected .mp4 extension, saved as: {files[0]}")

    def test_empty_result_flash_warning(self):
        """When all models return empty text, a warning flash must be present."""
        fake_audio = b'RIFF' + b'\x00' * 100
        mock_result = {
            'text': '',
            'segments': [],
            'duration': 1.0,
            'confidence': 0.0,
            'processing_time': 0.5,
            'language': 'kk',
        }
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result

        with patch.object(self.flask_app, '_get_transcriber', return_value=mock_transcriber), \
             patch('modules.audio_processor.AudioProcessor.prepare_audio',
                   return_value='/tmp/fake.wav'):
            response = self.client.post(
                '/transcribe',
                content_type='multipart/form-data',
                data={
                    'file': (io.BytesIO(fake_audio), 'speech.wav'),
                    'models': ['whisper_base'],
                    'source_language': 'kk',
                },
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        # Warning banner must appear on the page
        self.assertIn('вернули пустой результат', response.get_data(as_text=True))



# ---------------------------------------------------------------------------
# Translation module tests
# ---------------------------------------------------------------------------

class TestMarianTranslator(unittest.TestCase):
    """Unit tests for MarianTranslator that mock the HuggingFace models."""

    def _make_translator(self):
        from modules.translator import MarianTranslator
        return MarianTranslator()

    def test_supported_sources(self):
        from modules.translator import MarianTranslator
        sources = MarianTranslator.supported_sources()
        self.assertIn('en', sources)
        self.assertIn('ru', sources)

    def test_translate_empty_returns_empty(self):
        t = self._make_translator()
        self.assertEqual(t.translate('', 'en'), '')
        self.assertEqual(t.translate('  ', 'ru'), '')

    def test_translate_unsupported_pair_returns_empty(self):
        t = self._make_translator()
        result = t.translate('hello', src='de')  # German not supported
        self.assertEqual(result, '')

    def test_translate_en_to_kk_calls_model(self):
        """translate() must call the MarianMT model and return decoded output."""
        t = self._make_translator()

        fake_tokenizer = MagicMock()
        fake_tokenizer.return_value = {'input_ids': MagicMock()}
        fake_tokenizer.decode.return_value = 'Сәлем'

        fake_model = MagicMock()
        fake_model.generate.return_value = [[0]]  # fake token ids

        # Pre-populate the cache so no real download happens
        t._cache[('en', 'kk')] = (fake_model, fake_tokenizer)

        result = t.translate('Hello world', src='en')
        self.assertEqual(result, 'Сәлем')
        fake_model.generate.assert_called_once()
        # Verify the correct input text was forwarded to the tokenizer
        call_args = fake_tokenizer.call_args
        self.assertEqual(call_args[0][0], 'Hello world')

    def test_split_text_short(self):
        from modules.translator import _split_text
        chunks = _split_text('one two three')
        self.assertEqual(chunks, ['one two three'])

    def test_split_text_long(self):
        from modules.translator import _split_text
        words = ['word'] * 250
        chunks = _split_text(' '.join(words), max_words=100)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 100)


# ---------------------------------------------------------------------------
# Database translation column tests
# ---------------------------------------------------------------------------

class TestDatabaseTranslationColumns(unittest.TestCase):
    """Verify that the translation and source_language columns are stored."""

    def setUp(self):
        import importlib
        import tempfile
        import config
        self._tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp.close()
        self._orig_path = config.DATABASE_PATH
        config.DATABASE_PATH = self._tmp.name
        import modules.database as db_module
        importlib.reload(db_module)
        self.db = db_module
        self.db.init_db()

    def tearDown(self):
        import config
        config.DATABASE_PATH = self._orig_path
        os.unlink(self._tmp.name)

    def test_save_with_translation(self):
        record_id = self.db.save_transcription(
            filename='en_audio.wav',
            model='faster_whisper',
            text='Hello world',
            duration=2.0,
            confidence=0.9,
            translation='Сәлем дүние',
            source_language='en',
        )
        record = self.db.get_transcription(record_id)
        self.assertEqual(record['translation'], 'Сәлем дүние')
        self.assertEqual(record['source_language'], 'en')

    def test_save_without_translation_defaults_to_none(self):
        record_id = self.db.save_transcription(
            filename='kk_audio.wav',
            model='whisper_base',
            text='Сәлеметсіз бе',
            duration=1.0,
            confidence=0.8,
        )
        record = self.db.get_transcription(record_id)
        self.assertIsNone(record['translation'])
        self.assertIsNone(record['source_language'])

    def test_migrate_adds_columns_to_existing_db(self):
        """_migrate_db() must silently succeed on an already-migrated DB and
        must not destroy any pre-existing records."""
        # Insert a record before the second migration
        record_id = self.db.save_transcription(
            filename='pre_migration.wav',
            model='whisper_base',
            text='Тест',
            duration=1.0,
            confidence=0.8,
        )
        # Calling init_db() a second time (which calls _migrate_db() again)
        # must not raise even though the columns already exist.
        self.db.init_db()  # idempotent
        # Pre-existing record must survive the migration
        record = self.db.get_transcription(record_id)
        self.assertIsNotNone(record)
        self.assertEqual(record['text'], 'Тест')


# ---------------------------------------------------------------------------
# Flask app translation integration test
# ---------------------------------------------------------------------------

class TestTranslationInTranscribeRoute(unittest.TestCase):
    """Integration test: transcribe route with translate_to_kk flag."""

    def setUp(self):
        import importlib
        import tempfile
        import config
        self._tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp_db.close()
        self._tmp_upload = tempfile.mkdtemp()
        self._orig_db = config.DATABASE_PATH
        self._orig_upload = config.UPLOAD_FOLDER
        config.DATABASE_PATH = self._tmp_db.name
        config.UPLOAD_FOLDER = self._tmp_upload
        import modules.database as db_module
        importlib.reload(db_module)
        import app as flask_app
        importlib.reload(flask_app)
        flask_app.app.config['TESTING'] = True
        self.client = flask_app.app.test_client()
        self.flask_app = flask_app

    def tearDown(self):
        import config
        import shutil
        config.DATABASE_PATH = self._orig_db
        config.UPLOAD_FOLDER = self._orig_upload
        os.unlink(self._tmp_db.name)
        shutil.rmtree(self._tmp_upload, ignore_errors=True)

    def test_translation_stored_when_flag_set(self):
        """When translate_to_kk=1 and source is English, translation is saved."""
        fake_audio = b'RIFF' + b'\x00' * 100

        mock_transcribe_result = {
            'text': 'Hello world',
            'segments': [],
            'duration': 2.0,
            'confidence': 0.9,
            'processing_time': 1.0,
            'language': 'en',
        }
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_transcribe_result

        mock_translator = MagicMock()
        mock_translator.translate.return_value = 'Сәлем дүние'

        with patch.object(self.flask_app, '_get_transcriber', return_value=mock_transcriber), \
             patch('modules.audio_processor.AudioProcessor.prepare_audio',
                   return_value='/tmp/fake.wav'), \
             patch('modules.translator.get_translator', return_value=mock_translator):
            response = self.client.post(
                '/transcribe',
                content_type='multipart/form-data',
                data={
                    'file': (io.BytesIO(fake_audio), 'speech.wav'),
                    'models': ['whisper_base'],
                    'source_language': 'en',
                    'translate_to_kk': '1',
                },
            )

        self.assertIn(response.status_code, (200, 302))
        mock_translator.translate.assert_called_once_with('Hello world', src='en')

    def test_no_translation_when_source_is_kazakh(self):
        """When source language is Kazakh, the translator must NOT be called."""
        fake_audio = b'RIFF' + b'\x00' * 100

        mock_transcribe_result = {
            'text': 'Сәлеметсіз бе',
            'segments': [],
            'duration': 2.0,
            'confidence': 0.9,
            'processing_time': 1.0,
            'language': 'kk',
        }
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_transcribe_result

        mock_translator = MagicMock()

        with patch.object(self.flask_app, '_get_transcriber', return_value=mock_transcriber), \
             patch('modules.audio_processor.AudioProcessor.prepare_audio',
                   return_value='/tmp/fake.wav'), \
             patch('modules.translator.get_translator', return_value=mock_translator):
            self.client.post(
                '/transcribe',
                content_type='multipart/form-data',
                data={
                    'file': (io.BytesIO(fake_audio), 'speech.wav'),
                    'models': ['whisper_base'],
                    'source_language': 'kk',
                    'translate_to_kk': '1',
                },
            )

        mock_translator.translate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
