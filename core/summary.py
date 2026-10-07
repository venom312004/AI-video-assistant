import os
import re
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception,
)

MODEL_NAME = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
CHUNK_SIZE = 8000


def get_llm():
    return ChatGroq(
        model=MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_retries=0,   # tenacity handles retries
        timeout=60,
    )


def clean(text: str) -> str:
    """Remove Qwen3's <think>...</think> reasoning block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)  # opening tag missing
    return text.strip()


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


@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
def safe_invoke(chain, payload):
    try:
        return chain.invoke(payload)
    except Exception as e:
        print("=" * 60)
        print("GROQ API ERROR (SUMMARIZER)")
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


def _build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: x if isinstance(x, dict) else {"text": x})
        | prompt
        | get_llm()
        | StrOutputParser()
        | RunnableLambda(clean)
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    map_chain = _build_chain("Summarize this portion of a meeting transcript concisely.")

    if len(transcript) <= CHUNK_SIZE:
        chunks = [transcript]
    else:
        chunks = split_transcript(transcript)

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}")
        summary = safe_invoke(map_chain, {"text": chunk})
        chunk_summaries.append(summary)
        if i < len(chunks) - 1:
            time.sleep(3)

    combined = "\n\n".join(chunk_summaries)

    combined_chain = _build_chain(
        "You are an expert meeting summarizer. Combine these partial summaries "
        "into one final professional meeting summary in bullet points."
    )

    time.sleep(3)
    return safe_invoke(combined_chain, {"text": combined})


def generate_title(transcript: str) -> str:
    title_chain = _build_chain(
        "Based on the meeting transcript, generate a short professional meeting title "
        "(max 10 words). Only return the title, nothing else."
    )

    time.sleep(3)
    title = safe_invoke(title_chain, {"text": transcript[:2000]})
    return title.strip().strip('"').strip("'")