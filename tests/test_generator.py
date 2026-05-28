# tests/test_generator.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from generator import generate_interview_qa


@pytest.mark.asyncio
async def test_generate_interview_qa_basic():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps([
        {
            "difficulty": "basic",
            "interviewer": "What is Kubernetes?",
            "candidate": "Well, Kubernetes is basically a container orchestration platform..."
        },
        {
            "difficulty": "intermediate",
            "interviewer": "Explain pods in K8s.",
            "candidate": "Sure, a pod is the smallest deployable unit..."
        },
    ])

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await generate_interview_qa(
        client=mock_client,
        model="gpt-4o",
        topic="kubernetes",
        language="zh",
        count=2,
        existing_questions=[],
    )

    assert len(result) == 2
    assert result[0]["difficulty"] == "basic"
    assert "interviewer" in result[0]
    assert "candidate" in result[0]


@pytest.mark.asyncio
async def test_generate_interview_qa_with_existing():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps([
        {
            "difficulty": "advanced",
            "interviewer": "How does K8s handle networking?",
            "candidate": "Great question, K8s uses a flat network model..."
        },
    ])

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await generate_interview_qa(
        client=mock_client,
        model="gpt-4o",
        topic="kubernetes",
        language="zh",
        count=1,
        existing_questions=["What is Kubernetes?", "Explain pods."],
    )

    assert len(result) == 1
    # Verify existing questions were passed to the prompt
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    prompt_text = str(messages)
    assert "What is Kubernetes?" in prompt_text


@pytest.mark.asyncio
async def test_generate_interview_qa_retries_on_invalid_json():
    bad_response = MagicMock()
    bad_response.choices = [MagicMock()]
    bad_response.choices[0].message.content = "not valid json"

    good_response = MagicMock()
    good_response.choices = [MagicMock()]
    good_response.choices[0].message.content = json.dumps([
        {"difficulty": "basic", "interviewer": "Q1?", "candidate": "A1."}
    ])

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[bad_response, good_response]
    )

    result = await generate_interview_qa(
        client=mock_client,
        model="gpt-4o",
        topic="test",
        language="zh",
        count=1,
        existing_questions=[],
    )

    assert len(result) == 1
    assert mock_client.chat.completions.create.call_count == 2
