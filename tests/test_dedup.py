import pytest
from unittest.mock import AsyncMock, MagicMock
from dedup import tfidf_filter, llm_filter, deduplicate


def test_tfidf_filter_no_duplicates():
    existing = ["What is Kubernetes?", "Explain pods."]
    new = ["How does networking work in K8s?", "What is a Service?"]
    pairs = tfidf_filter(existing, new, threshold=0.7)
    assert pairs == []  # No duplicates found


def test_tfidf_filter_detects_similar():
    existing = ["What is Kubernetes?", "Explain pods."]
    new = ["What is Kubernetes used for?", "How does networking work?"]
    pairs = tfidf_filter(existing, new, threshold=0.3)
    # "What is Kubernetes?" and "What is Kubernetes used for?" should be similar
    assert len(pairs) >= 1
    assert pairs[0][0] == 0  # existing index
    assert pairs[0][1] == 0  # new index


def test_tfidf_filter_empty_existing():
    new = ["What is K8s?", "How to deploy?"]
    pairs = tfidf_filter([], new, threshold=0.5)
    assert pairs == []


@pytest.mark.asyncio
async def test_llm_filter_marks_duplicate():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "yes"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    candidates = [(0, 0, 0.8)]  # existing_idx=0, new_idx=0, similarity=0.8
    new_questions = ["What is K8s used for?"]
    existing_questions = ["What is Kubernetes?"]

    duplicates = await llm_filter(mock_client, "gpt-4o", existing_questions, new_questions, candidates)
    assert 0 in duplicates  # new question index 0 is duplicate


@pytest.mark.asyncio
async def test_llm_filter_marks_unique():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "no"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    candidates = [(0, 0, 0.6)]
    new_questions = ["How does K8s networking work?"]
    existing_questions = ["What is Kubernetes?"]

    duplicates = await llm_filter(mock_client, "gpt-4o", existing_questions, new_questions, candidates)
    assert 0 not in duplicates


@pytest.mark.asyncio
async def test_deduplicate_full_flow():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "yes"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    existing = ["What is Kubernetes?"]
    new = ["What is K8s?", "How to deploy apps?"]

    unique_indices = await deduplicate(
        mock_client, "gpt-4o", existing, new, threshold=0.3
    )
    # "What is K8s?" is similar to "What is Kubernetes?" -> duplicate
    # "How to deploy apps?" is different -> unique
    assert 1 in unique_indices  # index 1 is unique
