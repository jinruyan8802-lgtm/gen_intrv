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
