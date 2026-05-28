import json
from models import QAPair, InterviewSession


def test_qapair_creation():
    qa = QAPair(
        id="test-id",
        index=1,
        difficulty="basic",
        question="What is K8s?",
        answer="It's a container orchestrator.",
        tts_status="pending",
        audio_question="",
        audio_answer="",
        created_at="2026-05-28T00:00:00",
    )
    assert qa.id == "test-id"
    assert qa.tts_status == "pending"


def test_qapair_to_dict():
    qa = QAPair(
        id="test-id",
        index=1,
        difficulty="basic",
        question="What is K8s?",
        answer="It's a container orchestrator.",
        tts_status="pending",
        audio_question="",
        audio_answer="",
        created_at="2026-05-28T00:00:00",
    )
    d = qa.to_dict()
    assert d["id"] == "test-id"
    assert d["difficulty"] == "basic"


def test_qapair_from_dict():
    d = {
        "id": "test-id",
        "index": 1,
        "difficulty": "basic",
        "question": "What is K8s?",
        "answer": "It's a container orchestrator.",
        "tts_status": "pending",
        "audio_question": "",
        "audio_answer": "",
        "created_at": "2026-05-28T00:00:00",
    }
    qa = QAPair.from_dict(d)
    assert qa.id == "test-id"


def test_interview_session_roundtrip():
    session = InterviewSession(
        topic="kubernetes",
        language="zh",
        created_at="2026-05-28T00:00:00",
        updated_at="2026-05-28T00:00:00",
        qa_pairs=[],
    )
    qa = QAPair(
        id="q1",
        index=1,
        difficulty="basic",
        question="Q1?",
        answer="A1.",
        tts_status="pending",
        audio_question="",
        audio_answer="",
        created_at="2026-05-28T00:00:00",
    )
    session.qa_pairs.append(qa)

    json_str = session.to_json()
    restored = InterviewSession.from_json(json_str)
    assert restored.topic == "kubernetes"
    assert len(restored.qa_pairs) == 1
    assert restored.qa_pairs[0].question == "Q1?"
