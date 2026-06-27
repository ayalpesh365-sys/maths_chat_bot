# Agentic Mathematical Derivation Assistant

This is a Streamlit web app for a derivation-focused mathematical AI agent.

## Features

- Explains mathematical ideas
- Derives formulas step by step
- Searches PDF/text source documents using RAG
- Uses SymPy tools for symbolic verification
- Plots Taylor approximations
- Connects derivations to physics applications
- Supports follow-up/cross-questioning

## Local run

```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from your GitHub repository.
4. Set the main file path to `app.py`.
5. In the app settings, add this secret:

```toml
OPENAI_API_KEY = "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"
```

6. Deploy the app.

## Important security note

Do not upload `.env` or your real API key to GitHub.
