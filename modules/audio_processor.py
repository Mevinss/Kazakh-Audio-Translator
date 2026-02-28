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
        base, _ = os.path.splitext(audio_path)
        # Always produce a WAV file so FFmpeg never has to guess the output
        # format (which fails when the input file has no extension, e.g. when
        # secure_filename strips Cyrillic characters from the original name).
        output_path = base + '_normalized.wav'

        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-af', 'loudnorm',
            '-acodec', 'pcm_s16le',
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

    def denoise_audio(self, audio_path: str) -> str:
        """Apply basic FFmpeg denoising."""
        base, _ = os.path.splitext(audio_path)
        output_path = base + '_denoised.wav'
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-af', 'afftdn',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.SAMPLE_RATE),
            '-ac', '1',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Denoised audio to %s", output_path)
            return output_path
        except subprocess.CalledProcessError as exc:
            logger.warning("FFmpeg denoise error, using normalized audio: %s", exc.stderr.decode())
            return audio_path

    def prepare_audio(self, input_path: str, denoise: bool = False) -> str:
        """Full pipeline: extract audio (if needed) and normalize it.

        If normalization produces an empty or missing file it falls back to the
        pre-normalization audio so the models always receive a valid file.
        """
        _, ext = os.path.splitext(input_path)
        ext = ext.lower()
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

        if ext in video_extensions:
            audio_path = self.extract_audio(input_path)
        else:
            audio_path = input_path

        normalized = self.normalize_audio(audio_path)
        prepared = self.denoise_audio(normalized) if denoise else normalized

        # Guard against ffmpeg producing a zero-byte or missing output file,
        # which would make every model return empty text silently.
        if not os.path.isfile(prepared) or os.path.getsize(prepared) == 0:
            logger.warning(
                "Prepared file is missing or empty (%s); falling back to %s",
                prepared, audio_path,
            )
            return audio_path

        return prepared
