# Interview Podcast Generator - Design Spec

## Overview

A Python CLI tool that generates technical interview Q&A in conversational style, then synthesizes them into a dual-voice podcast MP3. Supports incremental generation with semantic deduplication to avoid repeating questions and audio.

## Goals

1. Generate 10-15 interview Q&A pairs per topic, from easy to hard
2. Q&A style should be natural/conversational, not textbook-like
3. Dual-voice podcast: interviewer + candidate, like a real interview
4. Incremental: second run merges new questions with existing, avoids regenerating audio for known questions
5. Semantic dedup: detect similar questions even if worded differently
6. Output readable Markdown + structured JSON + final MP3

## Architecture

### Project Structure

```
gen_intrv/
├── main.py              # CLI entry point, argument parsing
├── config.py            # Configuration management (.env loading)
├── generator.py         # LLM-based interview Q&A generation
├── dedup.py             # Semantic deduplication (TF-IDF + LLM)
├── tts.py               # OpenAI TTS synthesis (async, concurrent)
├── merger.py            # Audio concatenation via pydub
├── models.py            # Data models (Question, QA pair, etc.)
├── requirements.txt     # Dependencies
├── .env.example         # Example environment config
└── output/
    └── <topic>/
        ├── interview.md      # Human-readable Markdown
        ├── data.json         # Structured state (source of truth)
        ├── audio/
        │   ├── q001_interviewer.mp3
        │   ├── q001_candidate.mp3
        │   └── ...
        └── podcast.mp3       # Final merged podcast
```

### Data Flow

```
User Input (topic, language)
    │
    ▼
┌─────────────┐
│ 1. Generate  │ ── LLM generates Q&A pairs as JSON
│    Q&A Text  │     (with existing questions as context)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 2. Dedup     │ ── TF-IDF coarse filter (threshold 0.5)
│    & Merge   │     + LLM fine judgment for borderline cases
└──────┬──────┘     Merge new Q&A into existing data.json
       │
       ▼
┌─────────────┐
│ 3. Save      │ ── Update data.json + regenerate interview.md
│    Text      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 4. TTS       │ ── Only synthesize NEW Q&A pairs
│    Synthesis │     Concurrent calls (semaphore-limited)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 5. Merge     │ ── Concatenate all audio segments
│    Audio     │     Insert pauses between segments
└──────┬──────┘
       │
       ▼
   podcast.mp3
```

## Components

### 1. Configuration (`config.py`)

Loads from `.env` file using `python-dotenv`. All settings have defaults.

```python
# .env configuration:
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
TEXT_MODEL=gpt-4o

TTS_API_KEY=sk-xxx            # Optional, defaults to OPENAI_API_KEY
TTS_BASE_URL=                 # Optional, defaults to OPENAI_BASE_URL
TTS_INTERVIEWER_VOICE=alloy
TTS_CANDIDATE_VOICE=nova

LANGUAGE=zh                   # zh / en
QUESTION_COUNT=10
TTS_CONCURRENCY=3

DEDUP_TFIDF_THRESHOLD=0.5
```

Validation: API key missing → error at startup with clear message.

### 2. Data Model (`models.py`)

```python
@dataclass
class QAPair:
    id: str                    # UUID
    index: int                 # Sequence number
    difficulty: str            # "basic" / "intermediate" / "advanced"
    question: str              # Interviewer's question text
    answer: str                # Candidate's answer text
    tts_status: str            # "pending" / "done" / "failed"
    audio_question: str        # Path to question audio file
    audio_answer: str          # Path to answer audio file
    created_at: str            # ISO timestamp

@dataclass
class InterviewSession:
    topic: str
    language: str
    created_at: str
    updated_at: str
    qa_pairs: list[QAPair]
```

`data.json` serializes `InterviewSession` as the source of truth.

### 3. Text Generator (`generator.py`)

- Uses OpenAI-compatible API with configurable `base_url`
- System prompt enforces conversational style with filler words ("嗯", "其实", "well", "you know")
- Output: JSON array of `{difficulty, interviewer, candidate}`
- Difficulty levels: basic (3-4), intermediate (3-4), advanced (3-4)
- Incremental mode: passes existing question list to prompt, asks for different angles
- Response parsing: extract JSON from LLM response, validate structure

### 4. Semantic Dedup (`dedup.py`)

**Stage 1 - Coarse Filter (TF-IDF)**:
- Build TF-IDF vectors for all existing questions + new questions
- Compute pairwise cosine similarity
- Pairs with similarity > 0.5 → candidate for duplicate
- Pairs with similarity <= 0.5 → definitely new

**Stage 2 - Fine Filter (LLM)**:
- For each candidate duplicate pair, ask LLM: "Do these two questions test the same knowledge point? Answer yes or no."
- yes → duplicate, keep existing question
- no → new question, add to collection

**Merge Logic**:
- Existing questions and answers preserved as-is
- New questions appended with incremented index
- `data.json` and `interview.md` updated
- Only new Q&A pairs get `tts_status: "pending"`

### 5. TTS Synthesizer (`tts.py`)

- Async implementation using `openai` library
- Each QAPair generates two audio files: question (interviewer voice) + answer (candidate voice)
- Concurrency controlled by `asyncio.Semaphore` (default 3)
- Retry: max 2 retries per segment on failure
- Failed segments: mark `tts_status: "failed"` in data.json, don't block others
- File naming: `audio/q{index:03d}_interviewer.mp3`, `audio/q{index:03d}_candidate.mp3`
- Only processes pairs where `tts_status != "done"`

### 6. Audio Merger (`merger.py`)

- Uses `pydub` to concatenate all audio segments
- Insert 0.5s silence between question and answer within same pair
- Insert 1.0s silence between different Q&A pairs
- Output: `podcast.mp3` at 128kbps
- Rebuilds full audio from scratch each time (cheap operation, ensures correct order)

### 7. CLI Entry Point (`main.py`)

```bash
# Generate everything (text + audio)
python main.py kubernetes

# Specify language
python main.py kubernetes --lang en

# Text only, skip TTS
python main.py kubernetes --text-only

# Audio only (requires existing data.json)
python main.py kubernetes --audio-only

# Custom question count
python main.py kubernetes --count 15
```

Behavior:
- If `output/<topic>/data.json` exists → incremental mode (generate new, dedup, merge)
- If not exists → fresh generation
- `--text-only` skips TTS and merger steps
- `--audio-only` skips text generation, only runs TTS for pending items + merger

## Dependencies

```
openai>=1.0
python-dotenv
pydub
scikit-learn
```

System dependency: `ffmpeg` (required by pydub for MP3 handling)

## Error Handling

- Missing API key → exit with clear error message at startup
- LLM generation failure → retry once, then exit with error
- TTS segment failure → log, mark in data.json, continue with others
- Missing ffmpeg → detect at startup, prompt user to install
- Invalid JSON from LLM → retry with stricter prompt, max 2 attempts
- Empty topic or invalid input → validate at CLI level, show usage

## Markdown Output Format

```markdown
# Kubernetes 面试问答

> 生成时间: 2026-05-28 | 语言: 中文 | 问答数: 12

---

## 基础

### Q1: 什么是 Kubernetes？能简单介绍一下吗？

**面试官**: 什么是 Kubernetes？能简单介绍一下吗？

**候选人**: 嗯，简单来说，Kubernetes 就是一个容器编排工具...

---

### Q2: ...

---

## 进阶

### Q5: ...

---

## 深入

### Q9: ...
```
