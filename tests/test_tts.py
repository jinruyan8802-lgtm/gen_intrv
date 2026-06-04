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


@pytest.mark.asyncio
async def test_synthesize_single_edge_success(tmp_path):
    """Edge TTS success on first attempt."""
    output_path = str(tmp_path / "test.mp3")

    mock_communicate = MagicMock()

    async def mock_save(path):
        with open(path, "wb") as f:
            f.write(b"fake-edge-audio")

    mock_communicate.save = mock_save

    with patch("edge_tts.Communicate", return_value=mock_communicate) as mock_cls:
        result = await synthesize_single(
            None, "你好", "onyx", output_path,
            tts_provider="edge",
        )

    assert result is True
    assert os.path.exists(output_path)
    mock_cls.assert_called_once_with("你好", "zh-CN-YunjianNeural")


@pytest.mark.asyncio
async def test_synthesize_single_edge_retries_then_success(tmp_path):
    """Edge TTS retries on NoAudioReceived, then succeeds."""
    output_path = str(tmp_path / "test.mp3")

    mock_communicate = MagicMock()
    call_count = 0

    async def mock_save(path):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # Simulate edge-tts NoAudioReceived: file is empty after save
            with open(path, "wb") as f:
                f.write(b"")
            return
        with open(path, "wb") as f:
            f.write(b"fake-edge-audio")

    mock_communicate.save = mock_save

    with patch("edge_tts.Communicate", return_value=mock_communicate) as mock_cls:
        with patch("tts.asyncio.sleep") as mock_sleep:
            result = await synthesize_single(
                None, "Hello", "alloy", output_path,
                tts_provider="edge",
            )

    assert result is True
    # Inner retry loop should attempt 3 times (2 empty + 1 success)
    assert call_count == 3
    # Outer retry loop should NOT be invoked for edge provider
    # (if it were, save would be called 3 * 3 = 9 times)
    assert mock_cls.call_count == 3


@pytest.mark.asyncio
async def test_synthesize_single_edge_all_retries_fail():
    """Edge TTS all 5 inner retries fail; outer loop must not re-enter."""
    mock_communicate = MagicMock()

    async def mock_save(path):
        # Always empty → NoAudioReceived path
        with open(path, "wb") as f:
            f.write(b"")

    mock_communicate.save = mock_save

    with patch("edge_tts.Communicate", return_value=mock_communicate) as mock_cls:
        with patch("tts.asyncio.sleep"):
            result = await synthesize_single(
                None, "Hello", "alloy", "/tmp/test_edge.mp3",
                tts_provider="edge",
            )

    assert result is False
    # Should only attempt 5 times (inner loop), not 15 (outer + inner)
    assert mock_cls.call_count == 5


@pytest.mark.asyncio
async def test_synthesize_batch_edge_adds_delay_between_requests(tmp_path):
    """Edge TTS batch adds delay between serial requests."""
    output_path_a = str(tmp_path / "a.mp3")
    output_path_b = str(tmp_path / "b.mp3")

    mock_communicate = MagicMock()

    async def mock_save(path):
        with open(path, "wb") as f:
            f.write(b"audio")

    mock_communicate.save = mock_save

    items = [
        {"text": "First", "voice": "alloy", "output": output_path_a},
        {"text": "Second", "voice": "nova", "output": output_path_b},
    ]

    with patch("edge_tts.Communicate", return_value=mock_communicate):
        with patch("tts.asyncio.sleep") as mock_sleep:
            results = await synthesize_batch(
                None, items,
                tts_provider="edge",
                concurrency=5,  # Should be reduced to 1 internally
            )

    assert all(results)
    # Should sleep between the two requests (not after the last one)
    delay_calls = [c for c in mock_sleep.call_args_list
                   if c.args and c.args[0] == 1.5]
    assert len(delay_calls) == 1, f"Expected 1 delay call, got {len(delay_calls)}: {mock_sleep.call_args_list}"
