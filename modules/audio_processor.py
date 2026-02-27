import os
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Extracts and pre-processes audio from audio/video files."""

    SAMPLE_RATE = 16000

    def extract_audio(self, input_path: str) -> str:
        """Extract audio from a video or audio file and return path to a 16 kHz WAV."""
        base, _ = os.path.splitext(input_path)
        output_path = base + '_extracted.wav'

        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.SAMPLE_RATE),
            '-ac', '1',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Extracted audio to %s", output_path)
            return output_path
        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg error: %s", exc.stderr.decode())
            raise RuntimeError(f"Audio extraction failed: {exc.stderr.decode()}") from exc

    def normalize_audio(self, audio_path: str) -> str:
        """Normalize audio volume using FFmpeg loudnorm filter."""
        base, ext = os.path.splitext(audio_path)
        output_path = base + '_normalized' + ext

        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-af', 'loudnorm',
            '-ar', str(self.SAMPLE_RATE),
            '-ac', '1',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Normalized audio to %s", output_path)
            return output_path
        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg normalization error: %s", exc.stderr.decode())
            raise RuntimeError(f"Audio normalization failed: {exc.stderr.decode()}") from exc

    def prepare_audio(self, input_path: str) -> str:
        """Full pipeline: extract audio (if needed) and normalize it."""
        _, ext = os.path.splitext(input_path)
        ext = ext.lower()
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

        if ext in video_extensions:
            audio_path = self.extract_audio(input_path)
        else:
            audio_path = input_path

        return self.normalize_audio(audio_path)
