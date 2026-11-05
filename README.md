# RAG Chatbot

A retrieval-augmented chatbot over your own documents.

| Concern | Technology |
| --- | --- |
| Orchestration | LlamaIndex |
| Text generation | Gemini API (`google-genai`) |
| Embeddings | HuggingFace `sentence-transformers` (runs locally) |
| Vector store | ChromaDB (persistent, on disk) |
| Interface | Streamlit |

Documents are loaded, split into overlapping chunks, embedded locally, stored in
ChromaDB, and retrieved at question time to ground a Gemini answer with citations.

> **Status:** project scaffold. The module boundaries, configuration, and UI are
> complete; the pipeline stages intentionally raise `NotImplementedError`.

## Requirements

- Python 3.12 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Internet access for the first run only, to download the embedding model

## Installation

From PowerShell in the project folder:

```powershell
cd D:\rag-chatbot

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use `cmd.exe` instead of PowerShell, the activation line is:

```cmd
.venv\Scripts\activate.bat
```

## Configuration

Copy the template and fill in your key:

```powershell
Copy-Item .env.example .env
notepad .env
```

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | yes | — | Gemini credentials |
| `GEMINI_MODEL` | no | `gemini-2.5-flash` | Generation model |
| `EMBEDDING_MODEL` | no | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `HF_TOKEN` | no | empty | Only needed for gated HF models |
| `DATA_DIR` | no | `data` | Where source documents live |
| `CHROMA_DIR` | no | `vectorstore` | Chroma persistence path |
| `CHROMA_COLLECTION` | no | `rag_documents` | Collection name |
| `CHUNK_SIZE` | no | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | no | `50` | Repeated tokens between chunks |
| `SIMILARITY_TOP_K` | no | `3` | Chunks retrieved per question |

`.env` is gitignored. Keys are read once by `config.py` and never hardcoded,
printed, or passed into the UI layer.

## Adding documents

Place `.pdf`, `.docx`, `.txt`, or `.md` files in `data/` (subfolders are scanned
recursively). Document content is gitignored so private files stay local.

## Running

```powershell
streamlit run app.py
```

Streamlit opens at <http://localhost:8501>. Use **Ingest documents** in the
sidebar to build the index, then ask questions in the chat panel.

## Project layout

```
rag-chatbot/
├── app.py                    Streamlit entry point — UI only
├── config.py                 Loads and validates .env into Settings
├── requirements.txt          Pinned dependencies
├── .env.example              Configuration template, no secrets
├── .gitignore                Excludes .env, .venv, data, vectorstore
├── README.md
├── data/                     Source documents (contents ignored)
├── vectorstore/              ChromaDB persistence (ignored)
└── rag_chatbot/
    ├── __init__.py
    ├── ingestion.py          Find files, load them as Documents
    ├── chunking.py           Split Documents into TextNodes
    ├── embeddings.py         Build the HuggingFace embed model
    ├── llm.py                Build the Gemini chat model
    ├── vector_store.py       Open/reset Chroma, build and load the index
    ├── retrieval.py          Retrieve chunks, compose a cited Answer
    ├── chatbot.py            Conversation state and system prompt
    └── pipeline.py           Wires the stages in dependency order
```

Each stage depends only on `config.Settings` and LlamaIndex types, never on
another stage — so a module can be replaced or tested without touching the rest.
`pipeline.py` is the single place that knows the full order of operations.

## Extending

Suggested implementation order, one stage at a time:

1. `ingestion.py` — readers and source metadata
2. `chunking.py` — `SentenceSplitter` with the configured geometry
3. `embeddings.py` — `HuggingFaceEmbedding`
4. `llm.py` — `GoogleGenAI`
5. `vector_store.py` — persistent Chroma client and index
6. `retrieval.py` — retriever plus grounded prompt
7. `chatbot.py` and `pipeline.py` — wire it up

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Missing required environment variable 'GEMINI_API_KEY'` | `.env` is absent or blank — copy `.env.example` and fill it in |
| `CHUNK_OVERLAP must be smaller than CHUNK_SIZE` | Correct the values in `.env` |
| First run stalls while downloading the model | Expected once; the weights are cached under `~/.cache/huggingface` |
| "No index yet" banner stays visible | Click **Ingest documents**, then reload the page |
| `Checkpoint not found` from Streamlit | Cache cleared mid-session; refresh the browser tab |

## Security notes

- Never commit `.env` or paste `GEMINI_API_KEY` into an issue or screenshot.
- Rotate the key at Google AI Studio if it is ever exposed.
- Uploaded documents and the generated index stay local; nothing is sent to
  Google except the retrieved context of an active question.
