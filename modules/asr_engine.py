import os
import time
import tempfile
import logging

from modules.normalizer import KazakhNormalizer
from modules.transcribers.faster_whisper import FasterWhisperTranscriber
from modules.transcribers.whisper_medium import WhisperMediumTranscriber

logger = logging.getLogger(__name__)

# Three high-accuracy models for Kazakh ASR comparison
MODEL_WHISPER_MEDIUM = "whisper_medium"
MODEL_FASTER_WHISPER = "faster_whisper"
MODEL_WHISPER_LARGE_V3 = "whisper_large_v3"

MODEL_DISPLAY_NAMES = {
    MODEL_WHISPER_MEDIUM: "Whisper Medium",
    MODEL_FASTER_WHISPER: "Faster-Whisper Large-v3",
    MODEL_WHISPER_LARGE_V3: "Whisper Large-v3",
}


class ASREngine:
    """Unified engine for model inference, chunking, normalization and MT."""

    def __init__(self):
        self._normalizer = KazakhNormalizer()
        self._models = {}

    def _get_model(self, model_key: str):
        if model_key not in self._models:
            if model_key == MODEL_FASTER_WHISPER:
                self._models[model_key] = FasterWhisperTranscriber()
            elif model_key in (MODEL_WHISPER_MEDIUM, MODEL_WHISPER_LARGE_V3):
                self._models[model_key] = WhisperMediumTranscriber()
            else:
                raise ValueError(f"Unknown model: {model_key}")
        return self._models[model_key]

    def _split_chunks(self, audio_path: str, chunk_seconds: int = 30):
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)
            chunk_ms = chunk_seconds * 1000
            chunk_dir = tempfile.mkdtemp(prefix="kazakh_chunks_")
            chunks = []
            for idx, start in enumerate(range(0, len(audio), chunk_ms)):
                chunk = audio[start:start + chunk_ms]
                chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.wav")
                chunk.export(chunk_path, format="wav")
                chunks.append((chunk_path, start / 1000.0))
            return chunks or [(audio_path, 0.0)]
        except Exception as exc:
            logger.warning("Chunking unavailable, using single pass: %s", exc)
            return [(audio_path, 0.0)]

    def process(self, model_key: str, audio_path: str, language: str = "kk",
                apply_normalization: bool = True) -> dict:
        transcriber = self._get_model(model_key)
        started_at = time.time()
        full_text = []
        full_segments = []
        detected_language = language

        for chunk_path, offset in self._split_chunks(audio_path):
            chunk_result = transcriber.transcribe(chunk_path, language=language)
            detected_language = chunk_result.get("language", detected_language)
            for segment in chunk_result.get("segments", []):
                full_segments.append({
                    "start": round(segment["start"] + offset, 2),
                    "end": round(segment["end"] + offset, 2),
                    "text": segment["text"].strip(),
                })
            if chunk_result.get("text"):
                full_text.append(chunk_result["text"].strip())

        raw_text = " ".join(part for part in full_text if part).strip()
        normalized_text = self._normalizer.normalize(raw_text, use_llm=apply_normalization)
        elapsed = round(time.time() - started_at, 2)

        return {
            "text_raw": raw_text,
            "text": normalized_text,
            "segments": full_segments,
            "processing_time": elapsed,
            "duration": round(full_segments[-1]["end"], 2) if full_segments else 0.0,
            "confidence": 0.0,
            "language": detected_language,
        }
