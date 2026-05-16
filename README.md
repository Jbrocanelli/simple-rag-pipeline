# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline exposed as an MCP server, allowing AI assistants like Claude to query your local documents, ingest new files, and answer questions with source citations.

## Stack

- **LangChain** — document loading, splitting, and chain orchestration
- **ChromaDB** — vector store for document embeddings (persisted locally)
- **HuggingFace** — local embedding model (`all-MiniLM-L6-v2`)
- **Groq** — LLM inference (`llama-3.3-70b-versatile`)
- **FastMCP** — MCP server exposing the pipeline as tools, resources, and prompts

## Architecture

```
documents (PDF, TXT, MD)
        ↓
   load & split
        ↓
  embed & store (ChromaDB)
        ↓
   MMR retrieval
        ↓
  Groq LLM (llama-3.3-70b)
        ↓
  answer + sources
```

## MCP Interface

### Tools
| Tool | Description |
|------|-------------|
| `ask(prompt)` | Query the knowledge base and return an answer with source citations |
| `add_document(file_path)` | Ingest a new file or directory into the knowledge base |

### Resources
| Resource | Description |
|----------|-------------|
| `documents://list` | List all documents currently indexed in the knowledge base |

### Prompts
| Prompt | Description |
|--------|-------------|
| `response_style(question, style)` | Query with a specific response style: `concise`, `in-depth`, or `simple` |

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```

3. Add your documents (PDF, TXT, or MD) to the `data/` directory.

## Running the MCP Server

```bash
uv run mcp.py
```

Then register it with Claude Code:

```bash
claude mcp add --transport http rag-pipeline http://localhost:8000/mcp
```

## Project Structure

```
src/
  ingestion.py    # document loading and splitting
  vectorstore.py  # ChromaDB setup and caching
  retriever.py    # LangChain chain with MMR retrieval
mcp.py            # MCP server (tools, resources, prompts)
data/             # place your documents here
```
