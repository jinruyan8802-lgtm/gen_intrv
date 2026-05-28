# tests/test_tts.py
import os
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
