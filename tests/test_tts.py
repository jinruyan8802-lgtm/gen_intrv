# tests/test_tts.py
import os
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tts import synthesize_single, synthesize_batch


@pytest.mark.asyncio
async def test_synthesize_single_success(tmp_path):
    mock_client = AsyncMock()
    mock_response = MagicMock()

    async def mock_stream_to_file(path):
        with open(path, "wb") as f:
            f.write(b"fake-audio-data")

    mock_response.stream_to_file = mock_stream_to_file
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    output_path = str(tmp_path / "test.mp3")
    result = await synthesize_single(mock_client, "Hello world", "alloy", output_path)

    assert result is True
    assert os.path.exists(output_path)


@pytest.mark.asyncio
async def test_synthesize_single_failure():
    mock_client = AsyncMock()
    mock_client.audio.speech.create = AsyncMock(side_effect=Exception("API error"))

    result = await synthesize_single(mock_client, "Hello", "alloy", "/tmp/test.mp3", max_retries=0)
    assert result is False


@pytest.mark.asyncio
async def test_synthesize_batch_respects_concurrency(tmp_path):
    mock_client = AsyncMock()
    mock_response = MagicMock()

    async def mock_stream_to_file(path):
        with open(path, "wb") as f:
            f.write(b"audio")

    mock_response.stream_to_file = mock_stream_to_file
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    items = [
        {"text": "Hello", "voice": "alloy", "output": str(tmp_path / "a.mp3")},
        {"text": "World", "voice": "nova", "output": str(tmp_path / "b.mp3")},
    ]
    results = await synthesize_batch(mock_client, items, concurrency=1)
    assert all(results)
    assert mock_client.audio.speech.create.call_count == 2


@pytest.mark.asyncio
async def test_synthesize_single_cosyvoice_success(tmp_path):
    """CosyVoice provider uses subprocess; mock it."""
    output_path = str(tmp_path / "test.mp3")

    # Create dummy model dir so _synthesize_cosyvoice passes the existence check
    model_dir = tmp_path / "pretrained_models" / "speech-01-hd"
    model_dir.mkdir(parents=True)

    with patch("tts.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # CosyVoice generates wav; we need a dummy wav for pydub to convert
        wav_path = tmp_path / "test.wav"
        # Create a minimal valid WAV header (mono, 22050Hz, 16bit)
        import wave
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00" * 44100)  # 1 second of silence

        result = await synthesize_single(
            None, "你好", "alloy", output_path,
            tts_provider="cosyvoice",
            tts_model="speech-01-hd",
            cosyvoice_root=str(tmp_path),  # dummy path, subprocess is mocked
        )

    assert result is True
    assert os.path.exists(output_path)
    assert Path(output_path).stat().st_size > 0


@pytest.mark.asyncio
async def test_synthesize_single_cosyvoice_failure():
    """CosyVoice subprocess failure should return False."""
    with patch("tts.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="model not found")

        result = await synthesize_single(
            None, "你好", "alloy", "/tmp/test.mp3",
            tts_provider="cosyvoice", max_retries=0,
            cosyvoice_root="/nonexistent",
        )

    assert result is False
