# Actionable items, decisions, questions — combined into ONE Gemini call

import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
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
    return "429" in error_text or "rate limit" in error_text or "resource_exhausted" in error_text


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | prompt | llm | StrOutputParser()
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
        print("GEMINI API ERROR (EXTRACTOR)")
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


def extract_all(transcript: str) -> dict:
    """Single Gemini call that returns action items, decisions, and questions together."""
    chain = build_chain(COMBINED_SYSTEM_PROMPT)
    raw = safe_invoke(chain, transcript)

    def _extract_section(text: str, start_tag: str, end_tag: str = None) -> str:
        pattern = re.escape(start_tag) + r"(.*?)" + (re.escape(end_tag) if end_tag else "$")
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    action_items = _extract_section(raw, "###ACTION_ITEMS###", "###KEY_DECISIONS###")
    key_decisions = _extract_section(raw, "###KEY_DECISIONS###", "###OPEN_QUESTIONS###")
    open_questions = _extract_section(raw, "###OPEN_QUESTIONS###")

    return {
        "action_items": action_items or "No action items found.",
        "key_decisions": key_decisions or "No key decisions found.",
        "open_questions": open_questions or "No open questions found.",
    }