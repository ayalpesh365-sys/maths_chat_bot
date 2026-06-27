import os
import glob
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Mathematical Derivation Agent",
    page_icon="📐",
    layout="wide"
)


# -----------------------------
# Load API key safely
# -----------------------------

load_dotenv()

# Local development: use .env
# Streamlit Cloud deployment: use app secrets
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    # st.secrets may not exist in every local environment
    pass

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is missing. Add it to your .env file locally, "
        "or add it in Streamlit Cloud → App → Settings → Secrets."
    )
    st.stop()


# -----------------------------
# Auto-build vector database
# -----------------------------

CHROMA_PATH = Path("vectorstore/chroma_db")
PDF_PATH = Path("data/pdfs")
DOCS_PATH = Path("data/docs")

if not PDF_PATH.exists():
    PDF_PATH.mkdir(parents=True, exist_ok=True)

if not DOCS_PATH.exists():
    DOCS_PATH.mkdir(parents=True, exist_ok=True)

has_vector_db = CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())

if not has_vector_db:
    st.info("Building the vector database for the first time. This may take a few minutes...")

    result = subprocess.run(
        [sys.executable, "ingest.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        st.error("Vector database creation failed.")
        st.code(result.stderr)
        st.stop()


# Import only after API key and vector DB are ready
from agent import run_agent


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.title("📐 Math Derivation Agent")

    st.markdown(
        """
        **This agent can:**
        - Explain mathematical ideas
        - Derive formulas step by step
        - Verify symbolic identities
        - Plot Taylor approximations
        - Connect maths to physics applications
        - Generate practice/research questions
        """
    )

    st.warning(
        "This app uses the developer's OpenAI API key. Use it for educational testing only."
    )

    st.divider()

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Example prompts")

    examples = {
        "Taylor series of e^x": "Derive the Taylor series for e^x step by step.",
        "Euler formula": "Derive Euler formula using Taylor series.",
        "cos(iθ) = cosh(θ)": "Derive cos(i theta) = cosh(theta) and verify it symbolically.",
        "Plot sin approximation": "Plot Taylor approximation for sin.",
        "Physics applications": "Explain how Taylor series is used in physics and generate research-style questions."
    }

    for label, prompt in examples.items():
        if st.button(label):
            st.session_state.pending_prompt = prompt


# -----------------------------
# Main page
# -----------------------------

st.title("Agentic Mathematical Derivation Assistant")
st.caption(
    "A derivation-focused AI agent using RAG, OpenAI, SymPy, and plotting tools."
)

st.markdown(
    """
    Ask a derivation question, then cross-question the agent.

    Example follow-up questions:
    - Why did you use that formula?
    - Can you explain step 3 again?
    - Can you verify that identity symbolically?
    - Can you connect this to physics?
    """
)


# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# -----------------------------
# Display old messages
# -----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User input
# -----------------------------

user_input = st.chat_input("Ask or cross-question the agent...")

if st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# -----------------------------
# Run agent
# -----------------------------

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Deriving step by step..."):
            try:
                answer = run_agent(st.session_state.messages)
                st.markdown(answer)
            except Exception as e:
                answer = f"An error occurred:\n\n```text\n{e}\n```"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# -----------------------------
# Show latest generated plot
# -----------------------------

plot_files = sorted(
    glob.glob("outputs/plots/*.png"),
    key=os.path.getmtime,
    reverse=True
)

if plot_files:
    st.divider()
    st.subheader("Latest generated plot")
    st.image(plot_files[0], use_container_width=True)


# -----------------------------
# Download conversation
# -----------------------------

if st.session_state.messages:
    conversation_text = ""

    for msg in st.session_state.messages:
        role = msg["role"].upper()
        content = msg["content"]
        conversation_text += f"{role}:\n{content}\n\n"

    st.sidebar.download_button(
        label="⬇️ Download chat",
        data=conversation_text,
        file_name="math_derivation_chat.txt",
        mime="text/plain"
    )
