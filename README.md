# RAG Pipeline

A Retrieval-Augmented Generation pipeline that answers questions based on your local documents.

## Stack

- **LangChain** — document loading, splitting, and chain orchestration
- **ChromaDB** — vector store for document embeddings
- **HuggingFace** — local embedding model (`all-MiniLM-L6-v2`)
- **Groq** — LLM inference (`llama-3.3-70b-versatile`)

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```

3. Add your PDF or TXT files to `data/pdf/` and `data/text_files/`

## Usage

```bash
uv run python app.py
```
