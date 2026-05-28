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
    config = Config.from_env()
    language = args.lang or config.language
    count = args.count or config.question_count
    topic = args.topic

    paths = build_output_paths(topic)

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

        raw_qa = await generate_interview_qa(
            client=llm_client,
            model=config.text_model,
            topic=topic,
            language=language,
            count=count,
            existing_questions=existing_questions,
        )
        logger.info(f"Generated {len(raw_qa)} new Q&A pairs")

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

        save_session(session, paths["data_json"])
        logger.info(f"Saved {len(session.qa_pairs)} total Q&A pairs to {paths['data_json']}")

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
                tts_provider=config.tts_provider,
                tts_api_key=config.tts_api_key,
                tts_base_url=config.tts_base_url,
                tts_model=config.tts_model,
            )

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
