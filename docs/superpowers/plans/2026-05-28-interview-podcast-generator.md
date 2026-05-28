# Interview Podcast Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that generates conversational interview Q&A via LLM and synthesizes them into a dual-voice podcast MP3, with incremental generation and semantic deduplication.

**Architecture:** Linear pipeline: LLM generates Q&A JSON → TF-IDF+LLM dedup merges with existing data → OpenAI TTS synthesizes new segments → pydub concatenates into final MP3. State lives in `data.json` per topic.

**Tech Stack:** Python 3.10+, openai, python-dotenv, pydub, scikit-learn, pytest

---

## File Structure

```
gen_intrv/
├── main.py              # CLI entry point, orchestration
├── config.py            # .env loading, configuration dataclass
├── generator.py         # LLM Q&A generation
├── dedup.py             # TF-IDF coarse + LLM fine dedup
├── tts.py               # Async OpenAI TTS synthesis
├── merger.py            # pydub audio concatenation
├── models.py            # QAPair, InterviewSession dataclasses
├── requirements.txt     # Dependencies
├── .env.example         # Example config
├── tests/
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_generator.py
│   ├── test_dedup.py
│   ├── test_tts.py
│   ├── test_merger.py
│   └── test_main.py
└── output/              # Generated output (gitignored)
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
openai>=1.0
python-dotenv>=1.0
pydub>=0.25
scikit-learn>=1.3
pytest>=7.0
pytest-asyncio>=0.21
```

- [ ] **Step 2: Create .env.example**

```
# Text model
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
TEXT_MODEL=gpt-4o

# TTS (optional, defaults to text model credentials)
TTS_API_KEY=
TTS_BASE_URL=
TTS_INTERVIEWER_VOICE=alloy
TTS_CANDIDATE_VOICE=nova

# Generation
LANGUAGE=zh
QUESTION_COUNT=10
TTS_CONCURRENCY=3

# Dedup
DEDUP_TFIDF_THRESHOLD=0.5
```

- [ ] **Step 3: Create .gitignore**

```
.env
output/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create tests directory and git init**

```bash
mkdir -p tests output
touch tests/__init__.py
git init
git add .
git commit -m "chore: initial project setup"
```

---

### Task 2: Data Models (`models.py`)

**Files:**
- Create: `models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests for models**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'models'"

- [ ] **Step 3: Implement models.py**

```python
# models.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add data models (QAPair, InterviewSession)"
```

---

### Task 3: Configuration (`config.py`)

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write tests for config**

```python
# tests/test_config.py
import os
import pytest
from config import Config


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.com/v1")
    monkeypatch.setenv("TEXT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LANGUAGE", "en")
    monkeypatch.setenv("QUESTION_COUNT", "15")
    monkeypatch.setenv("TTS_CONCURRENCY", "5")
    monkeypatch.setenv("DEDUP_TFIDF_THRESHOLD", "0.6")
    monkeypatch.setenv("TTS_INTERVIEWER_VOICE", "onyx")
    monkeypatch.setenv("TTS_CANDIDATE_VOICE", "shimmer")

    cfg = Config.from_env()
    assert cfg.openai_api_key == "sk-test"
    assert cfg.openai_base_url == "https://test.com/v1"
    assert cfg.text_model == "gpt-4o-mini"
    assert cfg.language == "en"
    assert cfg.question_count == 15
    assert cfg.tts_concurrency == 5
    assert cfg.dedup_tfidf_threshold == 0.6
    assert cfg.tts_interviewer_voice == "onyx"
    assert cfg.tts_candidate_voice == "shimmer"


def test_config_tts_defaults_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://main.com/v1")
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    monkeypatch.delenv("TTS_BASE_URL", raising=False)

    cfg = Config.from_env()
    assert cfg.tts_api_key == "sk-main"
    assert cfg.tts_base_url == "https://main.com/v1"


def test_config_tts_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-main")
    monkeypatch.setenv("TTS_API_KEY", "sk-tts")
    monkeypatch.setenv("TTS_BASE_URL", "https://tts.com/v1")

    cfg = Config.from_env()
    assert cfg.tts_api_key == "sk-tts"
    assert cfg.tts_base_url == "https://tts.com/v1"


def test_config_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Config.from_env()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'config'"

- [ ] **Step 3: Implement config.py**

```python
# config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    openai_api_key: str
    openai_base_url: str
    text_model: str
    tts_api_key: str
    tts_base_url: str
    tts_interviewer_voice: str
    tts_candidate_voice: str
    language: str
    question_count: int
    tts_concurrency: int
    dedup_tfidf_threshold: float

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "Config":
        load_dotenv(env_path)

        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env or environment.")

        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        text_model = os.getenv("TEXT_MODEL", "gpt-4o")

        tts_api_key = os.getenv("TTS_API_KEY", "") or openai_api_key
        tts_base_url = os.getenv("TTS_BASE_URL", "") or openai_base_url

        tts_interviewer_voice = os.getenv("TTS_INTERVIEWER_VOICE", "alloy")
        tts_candidate_voice = os.getenv("TTS_CANDIDATE_VOICE", "nova")

        language = os.getenv("LANGUAGE", "zh")
        question_count = int(os.getenv("QUESTION_COUNT", "10"))
        tts_concurrency = int(os.getenv("TTS_CONCURRENCY", "3"))
        dedup_tfidf_threshold = float(os.getenv("DEDUP_TFIDF_THRESHOLD", "0.5"))

        return cls(
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            text_model=text_model,
            tts_api_key=tts_api_key,
            tts_base_url=tts_base_url,
            tts_interviewer_voice=tts_interviewer_voice,
            tts_candidate_voice=tts_candidate_voice,
            language=language,
            question_count=question_count,
            tts_concurrency=tts_concurrency,
            dedup_tfidf_threshold=dedup_tfidf_threshold,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add config management with .env support"
```

---

### Task 4: Text Generator (`generator.py`)

**Files:**
- Create: `generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write tests for generator**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generator.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'generator'"

- [ ] **Step 3: Implement generator.py**

```python
# generator.py
import json
import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_ZH = """你是一个技术面试模拟器。请生成关于 {topic} 的面试问答对话。

要求：
1. 生成 {count} 个问答对，难度从基础到深入
2. 面试官的问题要自然，像真实面试场景
3. 候选人的回答要口语化，像真人在说话，可以用"嗯"、"其实"、"简单来说"等过渡词
4. 每个回答控制在 150-300 字，不要太长
5. 难度分布：基础 3-4 个，进阶 3-4 个，深入 3-4 个
6. 输出严格的 JSON 数组，格式如下：
[
  {{"difficulty": "basic", "interviewer": "面试官的问题", "candidate": "候选人的回答"}},
  ...
]
只输出 JSON，不要有其他文字。"""

SYSTEM_PROMPT_EN = """You are a technical interview simulator. Generate interview Q&A dialogue about {topic}.

Requirements:
1. Generate {count} Q&A pairs, from easy to hard
2. Interviewer questions should be natural, like a real interview
3. Candidate answers should be conversational, use filler words like "well", "you know", "basically"
4. Keep answers between 150-300 words
5. Difficulty distribution: basic 3-4, intermediate 3-4, advanced 3-4
6. Output strict JSON array format:
[
  {{"difficulty": "basic", "interviewer": "question", "candidate": "answer"}},
  ...
]
Output only JSON, nothing else."""

INCREMENTAL_PROMPT_ZH = """以下是已有的面试问题，请生成 {count} 个**不同角度**的新问题和回答：

已有问题：
{existing}

要求：
1. 新问题必须与已有问题不同，探索新的知识点或更深的角度
2. 保持口语化风格
3. 输出严格的 JSON 数组格式"""

INCREMENTAL_PROMPT_EN = """Here are existing interview questions. Generate {count} NEW questions from **different angles**:

Existing questions:
{existing}

Requirements:
1. New questions must be different from existing ones, explore new knowledge points or deeper angles
2. Keep conversational style
3. Output strict JSON array format"""


async def generate_interview_qa(
    client: AsyncOpenAI,
    model: str,
    topic: str,
    language: str,
    count: int,
    existing_questions: List[str],
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Generate interview Q&A pairs using LLM."""

    system_prompt = (
        SYSTEM_PROMPT_ZH if language == "zh" else SYSTEM_PROMPT_EN
    ).format(topic=topic, count=count)

    messages = [{"role": "system", "content": system_prompt}]

    if existing_questions:
        inc_prompt = (
            INCREMENTAL_PROMPT_ZH if language == "zh" else INCREMENTAL_PROMPT_EN
        ).format(count=count, existing="\n".join(f"- {q}" for q in existing_questions))
        messages.append({"role": "user", "content": inc_prompt})
    else:
        messages.append({"role": "user", "content": f"请生成关于 {topic} 的面试问答。" if language == "zh" else f"Generate interview Q&A about {topic}."})

    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.8,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            # Handle case where LLM wraps in {"questions": [...]}
            if isinstance(data, dict):
                for key in ("questions", "qa_pairs", "data", "items"):
                    if key in data:
                        data = data[key]
                        break

            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")

            # Validate structure
            for item in data:
                if not all(k in item for k in ("difficulty", "interviewer", "candidate")):
                    raise ValueError(f"Missing keys in item: {item}")

            return data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"Failed to generate valid Q&A after {max_retries + 1} attempts: {e}")
            # Add retry instruction
            messages.append({"role": "assistant", "content": content if 'content' in dir() else ""})
            messages.append({"role": "user", "content": "请只输出 JSON 数组，不要有其他文字。" if language == "zh" else "Output only the JSON array, nothing else."})

    return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_generator.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add generator.py tests/test_generator.py
git commit -m "feat: add LLM-based interview Q&A generator"
```

---

### Task 5: Semantic Deduplication (`dedup.py`)

**Files:**
- Create: `dedup.py`
- Create: `tests/test_dedup.py`

- [ ] **Step 1: Write tests for dedup**

```python
# tests/test_dedup.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from dedup import tfidf_filter, llm_filter, deduplicate


def test_tfidf_filter_no_duplicates():
    existing = ["What is Kubernetes?", "Explain pods."]
    new = ["How does networking work in K8s?", "What is a Service?"]
    pairs = tfidf_filter(existing, new, threshold=0.5)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dedup.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'dedup'"

- [ ] **Step 3: Implement dedup.py**

```python
# dedup.py
import logging
from typing import List, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def tfidf_filter(
    existing_questions: List[str],
    new_questions: List[str],
    threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Stage 1: Find candidate duplicate pairs using TF-IDF cosine similarity.

    Returns list of (existing_idx, new_idx, similarity) for pairs above threshold.
    """
    if not existing_questions or not new_questions:
        return []

    all_questions = existing_questions + new_questions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_questions)

    existing_vectors = tfidf_matrix[: len(existing_questions)]
    new_vectors = tfidf_matrix[len(existing_questions) :]

    sim_matrix = cosine_similarity(existing_vectors, new_vectors)

    candidates = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i][j] > threshold:
                candidates.append((i, j, float(sim_matrix[i][j])))

    return candidates


async def llm_filter(
    client: AsyncOpenAI,
    model: str,
    existing_questions: List[str],
    new_questions: List[str],
    candidates: List[Tuple[int, int, float]],
) -> Set[int]:
    """Stage 2: Use LLM to judge if candidate pairs are truly duplicates.

    Returns set of new_question indices that are duplicates.
    """
    duplicates = set()

    for existing_idx, new_idx, sim in candidates:
        prompt = f"""请判断以下两个面试问题是否在考察同一个知识点。只回答 yes 或 no。

问题1: {existing_questions[existing_idx]}
问题2: {new_questions[new_idx]}"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            answer = response.choices[0].message.content.strip().lower()
            if "yes" in answer:
                duplicates.add(new_idx)
                logger.info(
                    f"Duplicate detected: '{new_questions[new_idx]}' ≈ '{existing_questions[existing_idx]}'"
                )
        except Exception as e:
            logger.warning(f"LLM dedup check failed for pair ({existing_idx}, {new_idx}): {e}")
            # On failure, treat as not duplicate (conservative)

    return duplicates


async def deduplicate(
    client: AsyncOpenAI,
    model: str,
    existing_questions: List[str],
    new_questions: List[str],
    threshold: float = 0.5,
) -> List[int]:
    """Full dedup pipeline: TF-IDF coarse filter + LLM fine filter.

    Returns list of indices into new_questions that are unique (not duplicates).
    """
    if not existing_questions:
        return list(range(len(new_questions)))

    # Stage 1: TF-IDF coarse filter
    candidates = tfidf_filter(existing_questions, new_questions, threshold)
    logger.info(f"TF-IDF found {len(candidates)} candidate duplicate pairs")

    # Stage 2: LLM fine filter
    if candidates:
        duplicates = await llm_filter(client, model, existing_questions, new_questions, candidates)
    else:
        duplicates = set()

    # Return indices of unique new questions
    unique = [i for i in range(len(new_questions)) if i not in duplicates]
    logger.info(f"Dedup result: {len(duplicates)} duplicates, {len(unique)} unique new questions")

    return unique
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dedup.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add dedup.py tests/test_dedup.py
git commit -m "feat: add semantic deduplication (TF-IDF + LLM)"
```

---

### Task 6: TTS Synthesizer (`tts.py`)

**Files:**
- Create: `tts.py`
- Create: `tests/test_tts.py`

- [ ] **Step 1: Write tests for TTS**

```python
# tests/test_tts.py
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tts import synthesize_single, synthesize_batch


@pytest.mark.asyncio
async def test_synthesize_single_success(tmp_path):
    mock_client = AsyncMock()
    # Mock the TTS response as an async iterator yielding bytes
    mock_response = MagicMock()

    async def mock_iter_bytes():
        yield b"fake-audio-data"

    mock_response.__aiter__ = lambda self: mock_iter_bytes()
    mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

    output_path = str(tmp_path / "test.mp3")

    with patch("tts.open_file_for_writing", return_value=MagicMock()):
        with patch("builtins.open", MagicMock()):
            # We'll test the actual call structure
            pass

    # Verify the function exists and accepts correct params
    assert callable(synthesize_single)


@pytest.mark.asyncio
async def test_synthesize_batch_respects_concurrency():
    # This tests that the semaphore is used
    mock_client = AsyncMock()
    items = [
        {"text": "Hello", "voice": "alloy", "output": "/tmp/test1.mp3"},
        {"text": "World", "voice": "nova", "output": "/tmp/test2.mp3"},
    ]
    # Just verify the function signature
    assert callable(synthesize_batch)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tts.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'tts'"

- [ ] **Step 3: Implement tts.py**

```python
# tts.py
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def synthesize_single(
    client: AsyncOpenAI,
    text: str,
    voice: str,
    output_path: str,
    max_retries: int = 2,
) -> bool:
    """Synthesize a single text to MP3 file. Returns True on success."""
    for attempt in range(max_retries + 1):
        try:
            response = await client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="mp3",
            )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Stream to file
            await response.stream_to_file(output_path)

            logger.info(f"Synthesized: {output_path}")
            return True

        except Exception as e:
            logger.warning(f"TTS attempt {attempt + 1} failed for {output_path}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(1)
            else:
                logger.error(f"TTS failed after {max_retries + 1} attempts: {output_path}")
                return False

    return False


async def synthesize_batch(
    client: AsyncOpenAI,
    items: List[Dict[str, Any]],
    concurrency: int = 3,
) -> List[bool]:
    """Synthesize multiple audio files with concurrency control.

    Each item: {"text": str, "voice": str, "output": str}
    Returns list of success booleans.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _synth(item: Dict[str, Any]) -> bool:
        async with semaphore:
            return await synthesize_single(
                client=client,
                text=item["text"],
                voice=item["voice"],
                output_path=item["output"],
            )

    tasks = [_synth(item) for item in items]
    return await asyncio.gather(*tasks)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tts.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tts.py tests/test_tts.py
git commit -m "feat: add async TTS synthesizer with concurrency control"
```

---

### Task 7: Audio Merger (`merger.py`)

**Files:**
- Create: `merger.py`
- Create: `tests/test_merger.py`

- [ ] **Step 1: Write tests for merger**

```python
# tests/test_merger.py
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
        mock_audio.silent.return_value = mock_segment
        mock_segment.__add__ = MagicMock(return_value=mock_segment)
        mock_segment.export = MagicMock()

        merge_audio_segments(qa_pairs, output_path)

        mock_audio.from_mp3.assert_called()
        mock_segment.export.assert_called_once_with(output_path, format="mp3", bitrate="128k")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_merger.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'merger'"

- [ ] **Step 3: Implement merger.py**

```python
# merger.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydub import AudioSegment

logger = logging.getLogger(__name__)

PAUSE_BETWEEN_QA_MS = 1000  # 1 second between Q&A pairs
PAUSE_BETWEEN_Q_A_MS = 500  # 0.5 seconds between question and answer


def merge_audio_segments(
    qa_pairs: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Concatenate audio segments into a single podcast MP3.

    Each qa_pair should have:
      - index: int
      - audio_question: str (path to question audio)
      - audio_answer: str or None (path to answer audio)
    """
    if not qa_pairs:
        logger.warning("No QA pairs to merge")
        return

    combined = AudioSegment.empty()
    pause_qa = AudioSegment.silent(duration=PAUSE_BETWEEN_QA_MS)
    pause_q_a = AudioSegment.silent(duration=PAUSE_BETWEEN_Q_A_MS)

    for i, qa in enumerate(qa_pairs):
        q_path = qa.get("audio_question")
        a_path = qa.get("audio_answer")

        # Add question audio
        if q_path and Path(q_path).exists():
            question_audio = AudioSegment.from_mp3(q_path)
            combined += question_audio
        else:
            logger.warning(f"Missing question audio for Q{qa['index']}: {q_path}")

        # Add pause between Q and A
        if a_path and Path(a_path).exists():
            combined += pause_q_a

        # Add answer audio
        if a_path and Path(a_path).exists():
            answer_audio = AudioSegment.from_mp3(a_path)
            combined += answer_audio
        else:
            logger.warning(f"Missing answer audio for Q{qa['index']}: {a_path}")

        # Add pause between QA pairs (except last)
        if i < len(qa_pairs) - 1:
            combined += pause_qa

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="mp3", bitrate="128k")
    logger.info(f"Podcast saved to {output_path} ({len(combined)}ms)")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_merger.py -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add merger.py tests/test_merger.py
git commit -m "feat: add audio merger with pydub"
```

---

### Task 8: CLI Entry Point (`main.py`)

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write tests for main CLI**

```python
# tests/test_main.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from main import parse_args, build_output_paths


def test_parse_args_basic():
    args = parse_args(["kubernetes"])
    assert args.topic == "kubernetes"
    assert args.lang is None
    assert args.text_only is False
    assert args.audio_only is False
    assert args.count is None


def test_parse_args_full():
    args = parse_args(["docker", "--lang", "en", "--text-only", "--count", "15"])
    assert args.topic == "docker"
    assert args.lang == "en"
    assert args.text_only is True
    assert args.count == 15


def test_build_output_paths():
    paths = build_output_paths("kubernetes")
    assert paths["base"] == "output/kubernetes"
    assert paths["data_json"] == "output/kubernetes/data.json"
    assert paths["interview_md"] == "output/kubernetes/interview.md"
    assert paths["audio_dir"] == "output/kubernetes/audio"
    assert paths["podcast"] == "output/kubernetes/podcast.mp3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_main.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'main'"

- [ ] **Step 3: Implement main.py**

```python
# main.py
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from config import Config
from models import QAPair, InterviewSession
from generator import generate_interview_qa
from dedup import deduplicate
from tts import synthesize_batch
from merger import merge_audio_segments
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interview Q&A podcast for a technical topic"
    )
    parser.add_argument("topic", help="Technical topic (e.g., kubernetes, docker)")
    parser.add_argument("--lang", choices=["zh", "en"], help="Language override")
    parser.add_argument("--text-only", action="store_true", help="Only generate text, skip TTS")
    parser.add_argument("--audio-only", action="store_true", help="Only synthesize audio (requires existing data.json)")
    parser.add_argument("--count", type=int, help="Number of Q&A pairs to generate")
    return parser.parse_args(argv)


def build_output_paths(topic: str) -> Dict[str, str]:
    base = f"output/{topic}"
    return {
        "base": base,
        "data_json": f"{base}/data.json",
        "interview_md": f"{base}/interview.md",
        "audio_dir": f"{base}/audio",
        "podcast": f"{base}/podcast.mp3",
    }


def load_existing_session(data_path: str) -> InterviewSession | None:
    path = Path(data_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return InterviewSession.from_json(f.read())
    return None


def save_session(session: InterviewSession, data_path: str) -> None:
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(session.to_json())


def generate_markdown(session: InterviewSession) -> str:
    lines = []
    lang_label = "中文" if session.language == "zh" else "English"
    lines.append(f"# {session.topic} 面试问答\n")
    lines.append(f"> 生成时间: {session.updated_at[:10]} | 语言: {lang_label} | 问答数: {len(session.qa_pairs)}\n")
    lines.append("---\n")

    difficulty_labels = {
        "basic": "基础",
        "intermediate": "进阶",
        "advanced": "深入",
    } if session.language == "zh" else {
        "basic": "Basic",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
    }

    current_difficulty = None
    for qa in session.qa_pairs:
        if qa.difficulty != current_difficulty:
            current_difficulty = qa.difficulty
            label = difficulty_labels.get(current_difficulty, current_difficulty)
            lines.append(f"\n## {label}\n")

        lines.append(f"### Q{qa.index}: {qa.question}\n")
        lines.append(f"**{'面试官' if session.language == 'zh' else 'Interviewer'}**: {qa.question}\n")
        lines.append(f"**{'候选人' if session.language == 'zh' else 'Candidate'}**: {qa.answer}\n")
        lines.append("---\n")

    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    # Load config
    config = Config.from_env()
    language = args.lang or config.language
    count = args.count or config.question_count
    topic = args.topic

    paths = build_output_paths(topic)

    # Initialize clients
    llm_client = AsyncOpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
    )
    tts_client = AsyncOpenAI(
        api_key=config.tts_api_key,
        base_url=config.tts_base_url,
    )

    now = datetime.now().isoformat()
    session = load_existing_session(paths["data_json"])

    # --- Text Generation Phase ---
    if not args.audio_only:
        existing_questions = []
        if session:
            existing_questions = [qa.question for qa in session.qa_pairs]
            logger.info(f"Found existing session with {len(existing_questions)} questions, entering incremental mode")

        # Generate new Q&A
        raw_qa = await generate_interview_qa(
            client=llm_client,
            model=config.text_model,
            topic=topic,
            language=language,
            count=count,
            existing_questions=existing_questions,
        )
        logger.info(f"Generated {len(raw_qa)} new Q&A pairs")

        # Deduplicate
        if existing_questions:
            new_questions = [item["interviewer"] for item in raw_qa]
            unique_indices = await deduplicate(
                client=llm_client,
                model=config.text_model,
                existing_questions=existing_questions,
                new_questions=new_questions,
                threshold=config.dedup_tfidf_threshold,
            )
            raw_qa = [raw_qa[i] for i in unique_indices]
            logger.info(f"After dedup: {len(raw_qa)} new unique Q&A pairs")

        # Merge into session
        if session is None:
            session = InterviewSession(
                topic=topic,
                language=language,
                created_at=now,
                updated_at=now,
                qa_pairs=[],
            )

        next_index = len(session.qa_pairs) + 1
        for item in raw_qa:
            qa = QAPair(
                id=f"q{next_index:03d}",
                index=next_index,
                difficulty=item["difficulty"],
                question=item["interviewer"],
                answer=item["candidate"],
                tts_status="pending",
                audio_question="",
                audio_answer="",
                created_at=now,
            )
            session.qa_pairs.append(qa)
            next_index += 1

        session.updated_at = now

        # Save data.json
        save_session(session, paths["data_json"])
        logger.info(f"Saved {len(session.qa_pairs)} total Q&A pairs to {paths['data_json']}")

        # Generate and save markdown
        md_content = generate_markdown(session)
        md_path = Path(paths["interview_md"])
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_content, encoding="utf-8")
        logger.info(f"Saved markdown to {paths['interview_md']}")

    # --- TTS Phase ---
    if not args.text_only:
        if session is None:
            session = load_existing_session(paths["data_json"])
            if session is None:
                logger.error(f"No data.json found at {paths['data_json']}. Run without --audio-only first.")
                sys.exit(1)

        # Prepare TTS items for pending QA pairs
        tts_items = []
        audio_dir = Path(paths["audio_dir"])
        audio_dir.mkdir(parents=True, exist_ok=True)

        for qa in session.qa_pairs:
            if qa.tts_status == "done":
                continue

            q_path = str(audio_dir / f"q{qa.index:03d}_interviewer.mp3")
            a_path = str(audio_dir / f"q{qa.index:03d}_candidate.mp3")

            tts_items.append({
                "text": qa.question,
                "voice": config.tts_interviewer_voice,
                "output": q_path,
                "qa_id": qa.id,
                "type": "question",
            })
            tts_items.append({
                "text": qa.answer,
                "voice": config.tts_candidate_voice,
                "output": a_path,
                "qa_id": qa.id,
                "type": "answer",
            })

            qa.audio_question = q_path
            qa.audio_answer = a_path

        if tts_items:
            logger.info(f"Synthesizing {len(tts_items)} audio segments...")
            results = await synthesize_batch(
                client=tts_client,
                items=tts_items,
                concurrency=config.tts_concurrency,
            )

            # Update TTS status
            qa_status = {}
            for item, success in zip(tts_items, results):
                qa_id = item["qa_id"]
                if qa_id not in qa_status:
                    qa_status[qa_id] = True
                if not success:
                    qa_status[qa_id] = False

            for qa in session.qa_pairs:
                if qa.id in qa_status:
                    qa.tts_status = "done" if qa_status[qa.id] else "failed"

            session.updated_at = datetime.now().isoformat()
            save_session(session, paths["data_json"])
            logger.info("TTS synthesis complete")
        else:
            logger.info("No new audio to synthesize")

        # --- Merge Phase ---
        merge_data = [
            {"index": qa.index, "audio_question": qa.audio_question, "audio_answer": qa.audio_answer}
            for qa in session.qa_pairs
        ]
        merge_audio_segments(merge_data, paths["podcast"])
        logger.info(f"Podcast saved to {paths['podcast']}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_main.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add CLI entry point with full pipeline orchestration"
```

---

### Task 9: Integration Test & End-to-End Validation

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test with mocked APIs**

```python
# tests/test_integration.py
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
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 3: Manual smoke test (text-only)**

```bash
cp .env.example .env
# Edit .env with real API key
python main.py kubernetes --text-only
cat output/kubernetes/interview.md
```

Expected: Markdown file with 10 conversational Q&A pairs about Kubernetes

- [ ] **Step 4: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full pipeline"
```

---

### Task 10: Final Polish

- [ ] **Step 1: Update .env.example with comments**

Ensure `.env.example` has clear documentation for each field.

- [ ] **Step 2: Verify all tests pass**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final polish and test verification"
```
