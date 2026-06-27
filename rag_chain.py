from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_PATH = "vectorstore/chroma_db"

# Must match ingest.py
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "math_derivation_sources"


def load_vector_db():
    if not Path(CHROMA_PATH).exists():
        raise FileNotFoundError("Vector DB not found. Run python ingest.py first.")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return db


def search_theory_notes(question: str) -> str:
    """
    Retrieve relevant chunks from PDF/text sources and include metadata.
    """
    db = load_vector_db()

    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant source chunks were retrieved."

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source_file = doc.metadata.get(
            "source_file",
            Path(doc.metadata.get("source", "unknown source")).name
        )
        page = doc.metadata.get("page", "unknown page")
        source_type = doc.metadata.get("source_type", "unknown")

        context_parts.append(
            f"""
[Source {i}]
Source file: {source_file}
Source type: {source_type}
Page: {page}

{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


def explain_from_notes(question: str) -> str:
    context = search_theory_notes(question)

    prompt = ChatPromptTemplate.from_template("""
You are a derivation-focused mathematical tutor.

Use the retrieved context to answer the question.

Rules:
- Focus on mathematical derivation.
- State assumptions.
- Show step-by-step logic.
- Do not jump directly to the final formula.
- Mention source file/page when useful.
- If the context is not enough, say what is missing.
- Connect to physics applications if relevant.

Question:
{question}

Retrieved context:
{context}

Answer:
""")

    model = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    chain = prompt | model

    response = chain.invoke({
        "question": question,
        "context": context
    })

    return response.content
