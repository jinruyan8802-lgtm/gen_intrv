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
        if q_path and Path(q_path).exists() and Path(q_path).stat().st_size > 0:
            question_audio = AudioSegment.from_mp3(q_path)
            combined += question_audio
        else:
            logger.warning(f"Missing question audio for Q{qa['index']}: {q_path}")

        # Add pause between Q and A
        if a_path and Path(a_path).exists() and Path(a_path).stat().st_size > 0:
            combined += pause_q_a

        # Add answer audio
        if a_path and Path(a_path).exists() and Path(a_path).stat().st_size > 0:
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
