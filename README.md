# Local Tax Document Q&A with Ollama

This project is a privacy-first, locally-run application for asking questions about your personal tax documents. It uses a hybrid RAG (Retrieval-Augmented Generation) pipeline, powered entirely by local models via Ollama, to provide answers based on your PDFs. No data ever leaves your machine.

The application is built with Python and features a Gradio web interface for easy interaction.

## Features

- **100% Local & Private**: All AI models (LLM and embeddings) run locally using Ollama. Your documents are processed on your machine and never sent to a third party.
- **Hybrid Search**: Combines semantic vector search (ChromaDB) and keyword-based search (BM25) for robust and accurate retrieval. Results are fused using Reciprocal Rank Fusion (RRF).
- **Structured Data Extraction**: Uses form-aware regex to extract key information from common tax forms, providing quick, direct answers to specific data points.
- **Gradio UI**: A simple and clean chat interface to upload documents, ask questions, and view results. Supports streaming responses.
- **Knowledge Base Integration**: Includes markdown files with general tax rules that are integrated into the RAG pipeline.
- **Efficient Ingestion**: Automatically detects if documents have changed to avoid re-processing, saving time on subsequent runs.

## Core Components

- **Backend**: Python, LangChain (for wrappers), Pydantic
- **UI**: Gradio
- **AI Models**:
  - **LLM**: `qwen3.5:9b` (or any other powerful local model)
  - **Embeddings**: `nomic-embed-text`
- **Vector Database**: ChromaDB (persistent on-disk)
- **Keyword Search**: `rank_bm25` (index persisted via pickle)
- **PDF Parsing**: `pdfplumber` and `PyMuPDF`

## How It Works

1. **Ingestion**:
    - You point the app to a folder containing your tax PDFs.
    - The system parses each PDF, extracting text and identifying tables.
    - Documents are split into page-aware, token-based chunks.
    - Chunks are converted into vector embeddings and stored in a local ChromaDB database.
    - A BM25 index is created for keyword search.
    - Structured data (like wages from a W-2) is extracted using regex and saved to JSON files.

2. **Querying**:
    - You ask a question in the Gradio chat interface.
    - The system performs a hybrid search: querying both ChromaDB (for semantic meaning) and the BM25 index (for keywords).
    - The results are combined and re-ranked.
    - The extracted structured data is also made available.
    - A prompt is constructed for the local LLM, containing your question and the retrieved context.
    - The LLM generates an answer, which is streamed back to the UI with citations to the source documents.

## Setup and Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running.
- The required Ollama models pulled:

    ```bash
    ollama pull qwen3.5:9b
    ollama pull nomic-embed-text
    ```

### Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd ollama-tax-doc-reader
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Install packages
    pip install -r requirements.txt
    ```

## How to Run

1. **Place your tax documents** (PDFs) into the `tax_docs/` directory.

2. **Start the Gradio application:**

    ```bash
    python app.py
    ```

3. **Open your browser** to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

4. The application will automatically start the ingestion process if it detects new or changed documents. You can also force a re-ingestion from the UI.

5. Once ingestion is complete, you can start asking questions in the chat interface.

## Configuration

The application's behavior can be customized via the `config.py` file or by setting environment variables.

Key settings include:

- `llm_model`: The Ollama model to use for chat.
- `embedding_model`: The Ollama model for embeddings.
- `chunk_size_tokens`: The size of text chunks for RAG.
- `vector_top_k` / `keyword_top_k`: The number of results to retrieve from each search method.
