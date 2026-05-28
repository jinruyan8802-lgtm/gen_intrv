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
