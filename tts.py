# tts.py
import asyncio
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)

# MiMo-V2.5-TTS voice mapping (friendly name -> voice_id)
MIMO_VOICE_MAP = {
    "alloy": "苏打",        # Chinese Male
    "nova": "茉莉",         # Chinese Female
    "onyx": "白桦",         # Chinese Male (mature)
    "shimmer": "冰糖",      # Chinese Female (default)
    "echo": "白桦",         # Chinese Male (deep)
    "fable": "茉莉",        # Chinese Female (warm)
}

# MiniMax voice ID mapping (friendly name -> voice_id)
MINIMAX_VOICE_MAP = {
    "alloy": "male-qn-qingse",
    "nova": "female-shaonv",
    "onyx": "male-qn-jingying",
    "shimmer": "female-yujie",
    "echo": "male-qn-badao",
    "fable": "female-chengshu",
}

# Edge-TTS voice mapping (friendly name -> edge voice)
EDGE_VOICE_MAP = {
    "alloy": "zh-CN-YunxiNeural",      # male, young
    "nova": "zh-CN-XiaoxiaoNeural",    # female, young
    "onyx": "zh-CN-YunjianNeural",     # male, mature
    "shimmer": "zh-CN-XiaoyiNeural",   # female, mature
    "echo": "zh-CN-YunjianNeural",     # male, deep
    "fable": "zh-CN-XiaochenNeural",   # female, warm
}


async def _synthesize_openai(
    client: AsyncOpenAI,
    text: str,
    voice: str,
    output_path: str,
) -> bool:
    """Synthesize using OpenAI-compatible TTS API."""
    response = await client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    await response.stream_to_file(output_path)
    if Path(output_path).stat().st_size == 0:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError("OpenAI TTS returned empty audio")
    return True


async def _synthesize_mimo(
    api_key: str,
    base_url: str,
    text: str,
    voice: str,
    output_path: str,
    model: str = "mimo-v2.5-tts",
) -> bool:
    """Synthesize using MiMo-V2.5-TTS API (OpenAI-compatible chat/completions)."""
    voice_id = MIMO_VOICE_MAP.get(voice, voice)

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "assistant", "content": text},
        ],
        "audio": {
            "format": "mp3",
            "voice": voice_id,
        },
    }

    async with httpx.AsyncClient(timeout=120) as http_client:
        resp = await http_client.post(url, headers=headers, json=payload)
        resp.raise_for_status()

        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"MiMo TTS: no choices in response")

        audio_obj = choices[0].get("message", {}).get("audio", {})
        audio_b64 = audio_obj.get("data", "")
        if not audio_b64:
            raise RuntimeError(f"MiMo TTS: no audio data in response")

        audio_bytes = base64.b64decode(audio_b64)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        if Path(output_path).stat().st_size == 0:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError("MiMo TTS returned empty audio")

    return True


async def _synthesize_minimax(
    api_key: str,
    base_url: str,
    text: str,
    voice: str,
    output_path: str,
    model: str = "speech-01-hd",
) -> bool:
    """Synthesize using MiniMax native TTS API (/v1/t2a_v2)."""
    voice_id = MINIMAX_VOICE_MAP.get(voice, voice)

    url = f"{base_url.rstrip('/')}/t2a_v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "text": text,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
        },
    }

    async with httpx.AsyncClient(timeout=60) as http_client:
        resp = await http_client.post(url, headers=headers, json=payload)
        resp.raise_for_status()

        data = resp.json()

        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax TTS error: {base_resp.get('status_msg', 'unknown')}")

        audio_hex = data.get("data", {}).get("audio", "")
        if not audio_hex:
            audio_hex = data.get("extra_info", {}).get("audio", "")
        if not audio_hex:
            raise RuntimeError(f"No audio data in MiniMax response: {list(data.keys())}")

        audio_bytes = bytes.fromhex(audio_hex)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        if Path(output_path).stat().st_size == 0:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError("MiniMax TTS returned empty audio")

    return True


async def _synthesize_edge(
    text: str,
    voice: str,
    output_path: str,
    language: str = "zh",
) -> bool:
    """Synthesize using edge-tts (free, no API key needed)."""
    import edge_tts

    voice_name = EDGE_VOICE_MAP.get(voice)
    if not voice_name:
        # Fallback: if voice is already a full edge voice name, use it directly
        if "-" in voice:
            voice_name = voice
        else:
            voice_name = "zh-CN-YunxiNeural" if language == "zh" else "en-US-GuyNeural"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(3):
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(output_path)

        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            return True

        logger.warning(f"Edge TTS returned empty audio, attempt {attempt + 1}/3")
        Path(output_path).unlink(missing_ok=True)
        if attempt < 2:
            await asyncio.sleep(1)

    raise RuntimeError("Edge TTS returned empty audio after 3 attempts")


async def synthesize_single(
    client: AsyncOpenAI,
    text: str,
    voice: str,
    output_path: str,
    max_retries: int = 2,
    tts_provider: str = "openai",
    tts_api_key: str = "",
    tts_base_url: str = "",
    tts_model: str = "speech-01-hd",
    language: str = "zh",
) -> bool:
    """Synthesize a single text to MP3 file. Returns True on success."""
    for attempt in range(max_retries + 1):
        try:
            if tts_provider == "mimo":
                await _synthesize_mimo(
                    api_key=tts_api_key,
                    base_url=tts_base_url,
                    text=text,
                    voice=voice,
                    output_path=output_path,
                    model=tts_model,
                )
            elif tts_provider == "minimax":
                await _synthesize_minimax(
                    api_key=tts_api_key,
                    base_url=tts_base_url,
                    text=text,
                    voice=voice,
                    output_path=output_path,
                    model=tts_model,
                )
            elif tts_provider == "edge":
                await _synthesize_edge(
                    text=text,
                    voice=voice,
                    output_path=output_path,
                    language=language,
                )
            else:
                await _synthesize_openai(client, text, voice, output_path)

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
    tts_provider: str = "openai",
    tts_api_key: str = "",
    tts_base_url: str = "",
    tts_model: str = "speech-01-hd",
    language: str = "zh",
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
                tts_provider=tts_provider,
                tts_api_key=tts_api_key,
                tts_base_url=tts_base_url,
                tts_model=tts_model,
                language=language,
            )

    tasks = [_synth(item) for item in items]
    return await asyncio.gather(*tasks)
