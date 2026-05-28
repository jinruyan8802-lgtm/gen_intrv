import json
import logging
import re
from typing import List, Dict, Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def extract_json_from_response(content: str) -> Any:
    """Extract JSON from LLM response, handling <think> tags and markdown code blocks."""
    # Remove <think>...</think> blocks (reasoning model output)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` code blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON array in the content
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {content[:200]}")

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
            )
            content = response.choices[0].message.content
            data = extract_json_from_response(content)

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
