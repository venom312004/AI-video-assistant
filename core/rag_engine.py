import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
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
    return "429" in error_text or "rate limit" in error_text or "resource_exhausted" in error_text


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
        print("GEMINI API ERROR (RAG)")
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


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = safe_invoke(rag_chain, question)
    print(f"answer: {answer}")
    return answer