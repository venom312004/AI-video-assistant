# Actionable items, decisions, questions — combined into ONE Groq (Qwen) call per chunk

import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

MODEL_NAME = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
CHUNK_CHARS = 12000  # ~3k tokens per chunk, lower this if you still hit 413/429


def get_llm():
    return ChatGroq(
        model=MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_retries=0,   # tenacity handles retries
        timeout=60,
    )


def is_retryable_error(exception):
    status_code = getattr(exception, "status_code", None)
    if status_code is None:
        response = getattr(exception, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    error_text = str(exception).lower()
    return "429" in error_text or "rate limit" in error_text or "rate_limit" in error_text


def clean(text: str) -> str:
    """Remove Qwen3's <think>...</think> reasoning block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)  # in case the opening tag is missing
    return text.strip()


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(clean)
    )


@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=2, min=3, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def safe_invoke(chain, payload):
    try:
        return chain.invoke(payload)
    except Exception as e:
        print("=" * 60)
        print("GROQ API ERROR (EXTRACTOR)")
        print("Exception type:", type(e).__name__)
        print("Error:", str(e))
        status_code = getattr(e, "status_code", None)
        if status_code is None:
            response = getattr(e, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)
        print("HTTP status:", status_code)
        print("=" * 60)
        raise


COMBINED_SYSTEM_PROMPT = """You are an expert meeting analyst. From the meeting transcript, extract all three of the following. Use EXACTLY these section headers so the output can be parsed:

###ACTION_ITEMS###
Numbered list of action items. For each: Task description, Owner (who is responsible), Deadline (if mentioned, else 'Not specified'). If none found, write "No action items found."

###KEY_DECISIONS###
Numbered list of key decisions made. If none found, write "No key decisions found."

###OPEN_QUESTIONS###
Numbered list of unresolved questions or topics needing follow-up. If none found, write "No open questions found."

Do not add any other text outside these three sections."""


def _extract_section(text: str, start_tag: str, end_tag: str = None) -> str:
    pattern = re.escape(start_tag) + r"(.*?)" + (re.escape(end_tag) if end_tag else r"$")
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _split_transcript(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on line/sentence boundaries so we don't cut mid-sentence."""
    if len(text) <= size:
        return [text]
    chunks, current = [], ""
    for part in re.split(r"(?<=[.!?\n])\s+", text):
        if len(current) + len(part) > size and current:
            chunks.append(current.strip())
            current = ""
        current += part + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _extract_one(chain, chunk: str) -> dict:
    raw = safe_invoke(chain, chunk)
    return {
        "action_items": _extract_section(raw, "###ACTION_ITEMS###", "###KEY_DECISIONS###"),
        "key_decisions": _extract_section(raw, "###KEY_DECISIONS###", "###OPEN_QUESTIONS###"),
        "open_questions": _extract_section(raw, "###OPEN_QUESTIONS###"),
    }


def _merge(sections: list[str], empty_msg: str) -> str:
    """Combine sections from all chunks, drop 'none found' lines, renumber."""
    items = []
    for sec in sections:
        for line in sec.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("no ") and "found" in line.lower():
                continue
            if re.match(r"^\d+[.)]\s+", line):
                items.append(re.sub(r"^\d+[.)]\s+", "", line))
            elif items:
                items[-1] += "\n   " + line  # continuation of previous item
            else:
                items.append(line)
    if not items:
        return empty_msg
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def extract_all(transcript: str) -> dict:
    """Returns action items, decisions, and questions. One call per chunk."""
    chain = build_chain(COMBINED_SYSTEM_PROMPT)
    results = [_extract_one(chain, c) for c in _split_transcript(transcript)]

    return {
        "action_items": _merge([r["action_items"] for r in results], "No action items found."),
        "key_decisions": _merge([r["key_decisions"] for r in results], "No key decisions found."),
        "open_questions": _merge([r["open_questions"] for r in results], "No open questions found."),
    }