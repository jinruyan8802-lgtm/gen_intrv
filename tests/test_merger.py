import pytest
from unittest.mock import MagicMock, patch
from merger import merge_audio_segments


def test_merge_audio_segments_calls_pydub(tmp_path):
    # Create dummy audio files
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create minimal MP3-like files (just for testing the merge logic)
    for name in ["q001_interviewer.mp3", "q001_candidate.mp3", "q002_interviewer.mp3"]:
        (audio_dir / name).write_bytes(b"\x00" * 100)

    qa_pairs = [
        {
            "index": 1,
            "audio_question": str(audio_dir / "q001_interviewer.mp3"),
            "audio_answer": str(audio_dir / "q001_candidate.mp3"),
        },
        {
            "index": 2,
            "audio_question": str(audio_dir / "q002_interviewer.mp3"),
            "audio_answer": None,  # Missing answer audio
        },
    ]

    output_path = str(tmp_path / "podcast.mp3")

    with patch("merger.AudioSegment") as mock_audio:
        mock_segment = MagicMock()
        mock_audio.from_mp3.return_value = mock_segment
        mock_audio.empty.return_value = mock_segment
        mock_audio.silent.return_value = mock_segment
        mock_segment.__iadd__ = MagicMock(return_value=mock_segment)
        mock_segment.__add__ = MagicMock(return_value=mock_segment)
        mock_segment.export = MagicMock()

        merge_audio_segments(qa_pairs, output_path)

        mock_audio.from_mp3.assert_called()
        mock_segment.export.assert_called_once_with(output_path, format="mp3", bitrate="128k")
