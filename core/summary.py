from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from tenacity import retry, wait_exponential, stop_after_attempt
import time
import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
        max_retries=6,      # let langchain retry internally too
        timeout=60,
    )

@retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(5))
def safe_invoke(chain, payload):
    """Wrapper: retries with growing backoff on 429s."""
    return chain.invoke(payload)

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)  # bigger chunks = fewer calls
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    # Skip chunking entirely for normal-length transcripts (most meetings)
    if len(transcript) <= 8000:
        chunks = [transcript]
    else:
        chunks = split_transcript(transcript)

    chunk_summaries = []
    for chunk in chunks:
        chunk_summaries.append(safe_invoke(map_chain, {"text": chunk}))
        time.sleep(1.2)  # stay under free-tier requests/sec limit

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting summarizer. Combine these partial summaries "
                   "into one final professional meeting summary in bullet points."),
        ("human", "{text}")
    ])
    combined_chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    return safe_invoke(combined_chain, combined)

def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the meeting transcript, generate a short professional meeting title "
                   "(max 10 words). Only return the title, nothing else."),
        ("human", "{text}")
    ])
    title_chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | title_prompt | llm | StrOutputParser()
    time.sleep(1)
    return safe_invoke(title_chain, transcript[:2000])