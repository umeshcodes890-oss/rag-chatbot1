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

> **Status:** ingestion and chunking are implemented. Embeddings, ChromaDB
> storage, retrieval, and Gemini generation are intentionally stubbed and raise
> `NotImplementedError`.

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
pip install -r requirements-dev.txt
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
├── requirements.txt          Pinned runtime dependencies
├── requirements-dev.txt      Test dependencies (pytest)
├── pytest.ini                Makes the project root importable in tests
├── .env.example              Configuration template, no secrets
├── .gitignore                Excludes .env, .venv, data, vectorstore
├── README.md
├── data/                     Source documents (contents ignored)
├── vectorstore/              ChromaDB persistence (ignored)
├── tests/
│   ├── conftest.py           Builds real PDF/DOCX/TXT fixtures
│   ├── test_ingestion.py     Cleaning, discovery, per-format loading, errors
│   └── test_chunking.py      Geometry, token budget, overlap, metadata
└── rag_chatbot/
    ├── __init__.py
    ├── ingestion.py          Find files, clean text, load them as Documents
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

## Testing

```cmd
.venv\Scripts\activate.bat
pytest
```

71 tests run offline in roughly 15 seconds. PDF and DOCX fixtures are generated
as real files rather than mocked, so `pypdf`, `docx2txt`, and the LlamaIndex
readers are genuinely exercised.

`tests/test_geminiapi.py` is excluded from collection on purpose. It is a
stand-alone script whose body calls the live Gemini API at import time, so
letting pytest import it would spend a real request every test run. Execute it
directly when you want a connectivity check:

```cmd
python tests\test_geminiapi.py
```

## Ingestion and chunking behaviour

- **Formats:** `.pdf` (one document per page via `PDFReader`), `.docx` (whole
  body via `DocxReader`), `.txt` and `.md` (direct UTF-8 read, latin-1 fallback).
- **Cleaning:** line endings normalise to `\n`, tabs and space runs collapse,
  trailing whitespace is stripped, three or more blank lines become one paragraph
  break, and zero-width characters are removed. Wording, casing, and punctuation
  are never rewritten.
- **Chunking:** `SentenceSplitter` at `chunk_size=512`, `chunk_overlap=50`,
  packing whole sentences so cuts land between words.
- **Empty and damaged input:** a file with no usable text contributes zero
  documents instead of raising; unreadable files raise `CorruptDocumentError`,
  unsupported types raise `UnsupportedFileError`, and absent paths raise
  `FileNotFoundError`. `load_documents(..., continue_on_error=True)` logs and
  skips those failures so one bad PDF cannot discard a whole knowledge base.
- **Metadata on every chunk:** `file_name`, `file_path`, `file_type`,
  `file_size_bytes`, `last_modified`, plus `page_number`, `page_count`, and
  `page_label` for PDFs. DOCX has no page fields because `docx2txt` discards
  pagination.

## Extending

Remaining stages, one at a time:

1. `embeddings.py` — `HuggingFaceEmbedding`
2. `llm.py` — `GoogleGenAI`
3. `vector_store.py` — persistent Chroma client and index
4. `retrieval.py` — retriever plus grounded prompt
5. `chatbot.py` and `pipeline.py` — wire it up

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
