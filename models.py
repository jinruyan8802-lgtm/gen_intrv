import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class QAPair:
    id: str
    index: int
    difficulty: str  # "basic" / "intermediate" / "advanced"
    question: str
    answer: str
    tts_status: str  # "pending" / "done" / "failed"
    audio_question: str
    audio_answer: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QAPair":
        return cls(**d)


@dataclass
class InterviewSession:
    topic: str
    language: str
    created_at: str
    updated_at: str
    qa_pairs: List[QAPair] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "language": self.language,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewSession":
        qa_pairs = [QAPair.from_dict(q) for q in d.get("qa_pairs", [])]
        return cls(
            topic=d["topic"],
            language=d["language"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            qa_pairs=qa_pairs,
        )

    @classmethod
    def from_json(cls, s: str) -> "InterviewSession":
        return cls.from_dict(json.loads(s))
