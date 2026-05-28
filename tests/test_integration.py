import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from main import run, parse_args


@pytest.mark.asyncio
async def test_full_pipeline_text_only(tmp_path, monkeypatch):
    """Test the full pipeline in text-only mode with mocked LLM."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.com/v1")
    monkeypatch.setenv("LANGUAGE", "zh")

    # Mock LLM response
    mock_qa_data = [
        {"difficulty": "basic", "interviewer": "What is K8s?", "candidate": "K8s is a container orchestrator."},
        {"difficulty": "intermediate", "interviewer": "Explain pods.", "candidate": "Pods are the smallest unit in K8s."},
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(mock_qa_data)

    with patch("main.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        # Run in text-only mode
        args = parse_args(["test-topic", "--text-only"])

        # Patch output path to use tmp
        with patch("main.build_output_paths") as mock_paths:
            paths = {
                "base": str(tmp_path / "test-topic"),
                "data_json": str(tmp_path / "test-topic" / "data.json"),
                "interview_md": str(tmp_path / "test-topic" / "interview.md"),
                "audio_dir": str(tmp_path / "test-topic" / "audio"),
                "podcast": str(tmp_path / "test-topic" / "podcast.mp3"),
            }
            mock_paths.return_value = paths

            await run(args)

    # Verify outputs
    assert Path(paths["data_json"]).exists()
    assert Path(paths["interview_md"]).exists()

    with open(paths["data_json"], "r") as f:
        data = json.load(f)
    assert data["topic"] == "test-topic"
    assert len(data["qa_pairs"]) == 2
    assert data["qa_pairs"][0]["difficulty"] == "basic"


@pytest.mark.asyncio
async def test_full_pipeline_incremental_mode(tmp_path, monkeypatch):
    """Test incremental mode: adding new Q&A to an existing session."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.com/v1")
    monkeypatch.setenv("LANGUAGE", "zh")

    # Create an existing session with one Q&A pair
    topic_dir = tmp_path / "test-topic"
    topic_dir.mkdir(parents=True, exist_ok=True)

    existing_session = {
        "topic": "test-topic",
        "language": "zh",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "qa_pairs": [
            {
                "id": "q001",
                "index": 1,
                "difficulty": "basic",
                "question": "What is Docker?",
                "answer": "Docker is a container platform.",
                "tts_status": "pending",
                "audio_question": "",
                "audio_answer": "",
                "created_at": "2025-01-01T00:00:00",
            }
        ],
    }
    (topic_dir / "data.json").write_text(json.dumps(existing_session, ensure_ascii=False, indent=2))

    # Mock LLM response with new Q&A pairs
    new_qa_data = [
        {"difficulty": "intermediate", "interviewer": "What are Docker volumes?", "candidate": "Volumes persist data beyond container lifecycle."},
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(new_qa_data)

    with patch("main.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        args = parse_args(["test-topic", "--text-only"])

        with patch("main.build_output_paths") as mock_paths:
            paths = {
                "base": str(topic_dir),
                "data_json": str(topic_dir / "data.json"),
                "interview_md": str(topic_dir / "interview.md"),
                "audio_dir": str(topic_dir / "audio"),
                "podcast": str(topic_dir / "podcast.mp3"),
            }
            mock_paths.return_value = paths

            await run(args)

    # Verify: session should now have both old and new Q&A pairs
    with open(paths["data_json"], "r") as f:
        data = json.load(f)
    assert data["topic"] == "test-topic"
    assert len(data["qa_pairs"]) == 2
    assert data["qa_pairs"][0]["question"] == "What is Docker?"
    assert data["qa_pairs"][1]["difficulty"] == "intermediate"


@pytest.mark.asyncio
async def test_full_pipeline_markdown_content(tmp_path, monkeypatch):
    """Test that generated markdown contains expected structure."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.com/v1")
    monkeypatch.setenv("LANGUAGE", "zh")

    mock_qa_data = [
        {"difficulty": "basic", "interviewer": "What is CI/CD?", "candidate": "CI/CD automates software delivery."},
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(mock_qa_data)

    with patch("main.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        args = parse_args(["ci-cd", "--text-only"])

        with patch("main.build_output_paths") as mock_paths:
            paths = {
                "base": str(tmp_path / "ci-cd"),
                "data_json": str(tmp_path / "ci-cd" / "data.json"),
                "interview_md": str(tmp_path / "ci-cd" / "interview.md"),
                "audio_dir": str(tmp_path / "ci-cd" / "audio"),
                "podcast": str(tmp_path / "ci-cd" / "podcast.mp3"),
            }
            mock_paths.return_value = paths

            await run(args)

    # Verify markdown structure
    md_content = Path(paths["interview_md"]).read_text(encoding="utf-8")
    assert "ci-cd" in md_content.lower() or "CI/CD" in md_content
    assert "Q1" in md_content
    assert "What is CI/CD?" in md_content
    assert "CI/CD automates software delivery." in md_content


@pytest.mark.asyncio
async def test_full_pipeline_json_wrapped_response(tmp_path, monkeypatch):
    """Test that LLM response wrapped in {'questions': [...]} is handled."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.com/v1")
    monkeypatch.setenv("LANGUAGE", "en")

    # Response wrapped in a dict with 'questions' key
    mock_qa_data = {
        "questions": [
            {"difficulty": "basic", "interviewer": "What is a class?", "candidate": "A class is a blueprint for objects."},
        ]
    }

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(mock_qa_data)

    with patch("main.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        args = parse_args(["oop", "--text-only", "--lang", "en"])

        with patch("main.build_output_paths") as mock_paths:
            paths = {
                "base": str(tmp_path / "oop"),
                "data_json": str(tmp_path / "oop" / "data.json"),
                "interview_md": str(tmp_path / "oop" / "interview.md"),
                "audio_dir": str(tmp_path / "oop" / "audio"),
                "podcast": str(tmp_path / "oop" / "podcast.mp3"),
            }
            mock_paths.return_value = paths

            await run(args)

    with open(paths["data_json"], "r") as f:
        data = json.load(f)
    assert data["language"] == "en"
    assert len(data["qa_pairs"]) == 1
    assert data["qa_pairs"][0]["question"] == "What is a class?"
