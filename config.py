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
    tts_provider: str  # "openai" or "minimax"
    tts_model: str     # e.g. "speech-01-hd" for minimax
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
        tts_provider = os.getenv("TTS_PROVIDER", "openai")  # "openai" or "minimax"
        tts_model = os.getenv("TTS_MODEL", "speech-01-hd")  # MiniMax model name

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
            tts_provider=tts_provider,
            tts_model=tts_model,
            language=language,
            question_count=question_count,
            tts_concurrency=tts_concurrency,
            dedup_tfidf_threshold=dedup_tfidf_threshold,
        )
