from langchain_mistralai import ChatMistralAI
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

import time
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
        max_retries=0,
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
    if "429" in error_text or "rate limit" in error_text:
        return True

    return False


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
        print("MISTRAL API ERROR")
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


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    if len(transcript) <= 8000:
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

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting summarizer. Combine these partial summaries "
                   "into one final professional meeting summary in bullet points."),
        ("human", "{text}")
    ])
    combined_chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()

    time.sleep(3)
    return safe_invoke(combined_chain, combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the meeting transcript, generate a short professional meeting title "
                   "(max 10 words). Only return the title, nothing else."),
        ("human", "{text}")
    ])
    title_chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | title_prompt | llm | StrOutputParser()

    time.sleep(3)
    return safe_invoke(title_chain, transcript[:2000])