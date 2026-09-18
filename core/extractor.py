# Actionable items, decisions, questions — combined into ONE Mistral call

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from tenacity import retry, wait_exponential, stop_after_attempt
import os
import re


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
        max_retries=6,
        timeout=60,
    )


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | prompt | llm | StrOutputParser()
    )


@retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(5))
def safe_invoke(chain, payload):
    return chain.invoke(payload)


COMBINED_SYSTEM_PROMPT = """You are an expert meeting analyst. From the meeting transcript, extract all three of the following. Use EXACTLY these section headers so the output can be parsed:

###ACTION_ITEMS###
Numbered list of action items. For each: Task description, Owner (who is responsible), Deadline (if mentioned, else 'Not specified'). If none found, write "No action items found."

###KEY_DECISIONS###
Numbered list of key decisions made. If none found, write "No key decisions found."

###OPEN_QUESTIONS###
Numbered list of unresolved questions or topics needing follow-up. If none found, write "No open questions found."

Do not add any other text outside these three sections."""


def extract_all(transcript: str) -> dict:
    """Single Mistral call that returns action items, decisions, and questions together."""
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