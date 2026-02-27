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
            def transcribe(self, audio_path: str) -> dict:
                return {'text': 'hello', 'segments': [], 'duration': 1.0, 'confidence': 1.0}

        t = DummyTranscriber()
        result = t.transcribe('fake.wav')
        self.assertEqual(result['text'], 'hello')


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


if __name__ == '__main__':
    unittest.main()
