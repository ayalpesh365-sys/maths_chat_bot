from dotenv import load_dotenv
from pathlib import Path
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

TEXT_PATH = "data/docs"
PDF_PATH = "data/pdfs"
CHROMA_PATH = "vectorstore/chroma_db"

# Keep this same in ingest.py and rag_chain.py.
EMBEDDING_MODEL = "text-embedding-3-small"

# Clear old vector database before rebuilding.
# This prevents mixing old Gemini vectors with new OpenAI vectors.
if Path(CHROMA_PATH).exists():
    shutil.rmtree(CHROMA_PATH)

Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

all_documents = []

# 1. Load text files
if Path(TEXT_PATH).exists():
    text_loader = DirectoryLoader(
        TEXT_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    text_documents = text_loader.load()

    for doc in text_documents:
        doc.metadata["source_type"] = "text"
        doc.metadata.setdefault("source_file", Path(doc.metadata.get("source", "unknown.txt")).name)

    all_documents.extend(text_documents)
    print(f"Loaded text documents: {len(text_documents)}")

# 2. Load PDF files
pdf_files = list(Path(PDF_PATH).glob("*.pdf"))

for pdf_file in pdf_files:
    print(f"Loading PDF: {pdf_file}")

    loader = PyPDFLoader(str(pdf_file))
    pdf_documents = loader.load()

    for doc in pdf_documents:
        doc.metadata["source_file"] = pdf_file.name
        doc.metadata["source_type"] = "pdf"

    all_documents.extend(pdf_documents)

print(f"Total documents/pages loaded: {len(all_documents)}")

if not all_documents:
    raise ValueError("No documents found. Add .txt files to data/docs or PDFs to data/pdfs.")

# 3. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(all_documents)
print(f"Created chunks: {len(chunks)}")

# 4. Create OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

# 5. Store in Chroma
_ = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name="math_derivation_sources"
)

print("Vector database created successfully using OpenAI embeddings.")
