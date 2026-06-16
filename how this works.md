# How This RAG Tax App Works — A Beginner's Guide

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [Big Picture: The Data Flow](#2-big-picture-the-data-flow)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
   - [config.py — All the Knobs and Dials](#configpy--all-the-knobs-and-dials)
   - [app.py — The Front Door (Gradio UI)](#apppy--the-front-door-gradio-ui)
   - [ingest.py — The Ingestion Boss](#ingestpy--the-ingestion-boss)
   - [pdf_parser.py — Reading PDFs](#pdf_parserpy--reading-pdfs)
   - [chunker.py — Slicing Text Into Bite-Size Pieces](#chunkerpy--slicing-text-into-bite-size-pieces)
   - [embeddings.py — Turning Words Into Numbers](#embeddingspy--turning-words-into-numbers)
   - [vector_store.py — The Semantic Memory](#vector_storepy--the-semantic-memory)
   - [keyword_search.py — The Old-School Index (BM25)](#keyword_searchpy--the-old-school-index-bm25)
   - [rag_engine.py — Fusing Two Worlds Together](#rag_enginepy--fusing-two-worlds-together)
   - [query_pipeline.py — The Brain That Answers Questions](#query_pipelinepy--the-brain-that-answers-questions)
   - [llm_client.py — Talking to Ollama](#llm_clientpy--talking-to-ollama)
   - [structured_extractor.py — Reading Tax Forms by Pattern](#structured_extractorpy--reading-tax-forms-by-pattern)
   - [tax_calculator.py — Doing the Math](#tax_calculatorpy--doing-the-math)
4. [Why Hybrid Search? (Vector + BM25)](#4-why-hybrid-search-vector--bm25)
5. [Why Structured Extraction on Top of RAG?](#5-why-structured-extraction-on-top-of-rag)
6. [How a Question Flows Through the System](#6-how-a-question-flows-through-the-system)
7. [Glossary of RAG Terms](#7-glossary-of-rag-terms)
8. [Key Design Decisions Summary](#8-key-design-decisions-summary)

---

## 1. What Are We Building?

This is a **local, private tax Q&A bot**. You give it your tax PDFs (W-2s, 1040 forms, etc.), it reads them all, stores them in a smart searchable format, and then you can ask natural-language questions like:

- "How much federal tax was withheld?"
- "What's my effective tax rate?"
- "Show me the Massachusetts tax details."

Everything runs on your own machine using **Ollama** — no data ever leaves your computer.

### Three Big Pieces

```
┌─────────────────────┐
│  1. INGESTION       │  Read PDFs → chunk → embed → store
│  (happens once)     │
├─────────────────────┤
│  2. RETRIEVAL       │  When you ask something, find the
│  (happens per Q)    │  most relevant chunks of text
├─────────────────────┤
│  3. GENERATION      │  Feed those chunks to an LLM,
│  (happens per Q)    │  which writes the answer
└─────────────────────┘
```

This is called **RAG** — **Retrieval-Augmented Generation**. Instead of asking the LLM to know everything from memory (which it won't for your personal tax docs), we *retrieve* the relevant information first and give it to the LLM as context. The LLM just has to read and summarize.

---

## 2. Big Picture: The Data Flow

```
─── INGESTION ──────────────────────────────────────────────────

  tax_docs/*.pdf ──→ pdf_parser.py ──→ chunker.py ──→ vector_store.py (ChromaDB)
                        │                    │              + embedding via Ollama
                        │                    └──→ keyword_search.py (BM25 index)
                        │
                        └──→ structured_extractor.py → tax_summary.json

─── QUESTION ANSWERING ─────────────────────────────────────────

  User Question
       │
       ▼
  query_pipeline.py
       │
       ├──→ Vector search (ChromaDB) ── top 8 chunks
       ├──→ Keyword search (BM25)     ── top 8 chunks
       │
       ├──→ Reciprocal Rank Fusion (merge both lists)
       │
       ├──→ Load structured data + optional calculator
       │
       ├──→ Assemble prompt: [system prompt + history + context + question]
       │
       └──→ LLM (Ollama / qwen3.5:9b) → stream answer → Gradio UI
```

---

## 3. File-by-File Breakdown

### config.py — All the Knobs and Dials

```python
class AppConfig(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embedding_model: str = "nomic-embed-text:latest"
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 200
    vector_top_k: int = 8       # how many results from vector search
    keyword_top_k: int = 8      # how many results from keyword search
    final_top_k: int = 10       # final merged results to give LLM
    llm_temperature: float = 0.1
    memory_turns: int = 5       # how many chat history turns to keep
```

**What it does:** Central place for every setting. Uses `pydantic-settings` so you can override any value with an environment variable prefixed `TAX_AI_` (e.g., `TAX_AI_LLM_MODEL=llama3`).

**Why this way:** Hard-coding settings in each file creates a maintenance mess. One config object gets passed everywhere. The env-var override is useful for Docker or CI without editing files.

---

### app.py — The Front Door (Gradio UI)

**What it does:** Creates the web interface and wires up three buttons:

| Button | What Happens |
|--------|-------------|
| **Ingest Documents** | Calls `ingest_folder()` — reads all PDFs, builds search indexes |
| **Show Structured Summary** | Loads `tax_summary.json` and displays extracted fields |
| **Send** (chat) | Streams a Q&A response from the LLM |

**Key pattern — streaming:** The chat uses Python generators (`yield`). This lets Gradio update the message token-by-token as the LLM responds, instead of making you wait for the full answer.

```python
# app.py:69-80 — the streaming loop
for event in query_engine.stream_answer(message, ...):
    if event.get("type") == "meta":
        # First event: metadata about sources found
        source_lines = ...
        continue
    token = event.get("content", "")
    reply_parts.append(token)
    history[-1]["content"] = "".join(reply_parts)
    yield history, ""  # Gradio updates the UI here
```

**Why this way:** Gradio's `yield`-based streaming is simple and works with any LLM that supports token-by-token streaming (which Ollama does).

---

### ingest.py — The Ingestion Boss

**What it does:** Orchestrates the whole "reading documents" pipeline.

**Step by step:**

1. **Signature check** — Computes a SHA-256 hash of the folder (PDF filenames + sizes + last-modified times). If nothing changed since last time, it skips re-ingestion entirely. This saves minutes of waiting when you restart the app.

2. **Discover PDFs** — Finds all `*.pdf` files in the folder (including subfolders).

3. **Parse PDFs** — Extracts text and tables from each PDF.

4. **Load knowledge docs** — Reads `knowledge/*.md` files as extra documents about tax rules.

5. **Chunk everything** — Splits all text into overlapping chunks of ~800 tokens each.

6. **Build vector store** — Embeds chunks and stores them in ChromaDB.

7. **Build BM25 index** — Creates a keyword search index and saves to disk.

8. **Extract structured data** — Runs regex patterns to pull out numerical fields.

9. **Save state** — Writes `ingestion_state.json` so next time it can skip.

**Why this way:** The signature check makes re-launching the app instant if documents haven't changed. Treating knowledge markdown files as regular documents (not hardcoded prompts) means you can add new tax rules without editing code.

---

### pdf_parser.py — Reading PDFs

**What it does:** Takes a PDF file path, returns a structured dictionary with all the text, tables, and metadata.

**Two PDF libraries used:**

| Library | When | Why |
|---------|------|-----|
| **pdfplumber** | Primary | Great text + table extraction; converts tables to markdown |
| **PyMuPDF (fitz)** | Fallback | Used on pages where pdfplumber extracted nothing |

**Table handling:** Tables are converted to markdown format with a `[TABLE]` marker so the chunker knows tables are present:

```
| Item | Amount |
|------|--------|
| Wages | 50000 |
```

**Document type detection:** The parser looks at filenames and first-page text to figure out what kind of form this is — W-2, 1099, 1040, Massachusetts Form 1, or India-US tax treaty. This metadata flows through to every chunk so the system knows what it's looking at.

**Why this way:** pdfplumber handles most PDFs well, but some scanned or unusual PDFs need PyMuPDF's fallback. The `[TABLE]` marker is a simple way to tell downstream components "this chunk has tabular data."

---

### chunker.py — Slicing Text Into Bite-Size Pieces

**What it does:** Takes a parsed document and splits it into overlapping chunks of text.

**Why chunk?** LLMs have a context window limit (how much text they can "see" at once). Chunking also lets retrieval find *specific* passages rather than entire documents.

```
A 10-page document (say 5000 tokens) might become 7 chunks:

  Chunk 1: tokens 0-800
  Chunk 2: tokens 600-1400  (200-token overlap with chunk 1)
  Chunk 3: tokens 1200-2000
  ...
```

**Overlap** is critical. If a sentence bridges across two chunks, overlap ensures both chunks have the complete sentence.

**Token counting:** Uses `tiktoken` with OpenAI's `cl100k_base` encoding. This counts tokens the same way LLMs do (not just word count).

```python
# chunker.py:17-33 — the core splitting logic
def _split_text_by_tokens(text, chunk_size, overlap):
    tokens = enc.encode(text)
    start = 0
    step = chunk_size - overlap  # how far to slide the window
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += step
```

**Why this way:** Fixed-token-size chunks are the simplest reliable approach. Overlap prevents cutting sentences in half. The sliding window means every token appears in roughly `chunk_size / step` chunks, so retrieval has multiple chances to find it.

---

### embeddings.py — Turning Words Into Numbers

**What it does:** A thin wrapper around Ollama's embedding model.

**The key concept:** An **embedding** is a list of numbers (a vector) that represents the *meaning* of a piece of text. Sentences with similar meanings get similar vectors.

```
"federal tax withholding" → [0.12, -0.45, 0.78, ...]  (768 numbers)
"I love pizza"            → [0.89, 0.23, -0.15, ...]  (very different vector)
```

**Cosine similarity:** The system measures how similar two vectors are by calculating the angle between them. Close angle = similar meaning.

**The model:** `nomic-embed-text:latest` via Ollama. It runs locally, so your documents never leave your machine.

**Two methods:**
- `embed_documents(texts)` — Used during ingestion to embed all chunks
- `embed_query(text)` — Used at question time to embed the user's question

**Why this way:** LangChain's `OllamaEmbeddings` handles all the HTTP communication with Ollama. We wrap it in our own class so the rest of the code doesn't depend on LangChain directly.

---

### vector_store.py — The Semantic Memory

**What it does:** Wraps ChromaDB, a vector database that stores embeddings and finds similar ones.

**What is a vector database?** Normal databases find exact matches ("where name = 'Smith'"). Vector databases find *similar* matches ("find text most like 'how much tax was withheld'").

**During ingestion:**
```python
vector_store.add_chunks(chunks)
# Internally: chunk.text → embedding vector → stored in ChromaDB
```

**During query:**
```python
vector_store.similarity_search_with_score(question, k=8)
# question → embedding vector → ChromaDB finds 8 nearest neighbors
# Returns: [(Document, similarity_score), ...]
```

**Persistence:** ChromaDB saves to `data/chroma_db/` on disk. The data survives app restarts.

**Why this way:** ChromaDB is lightweight, persists to disk, has a simple API, and works well with LangChain. It's the simplest production-quality vector DB for a local app.

---

### keyword_search.py — The Old-School Index (BM25)

**What it does:** Implements keyword-based search using the BM25 algorithm.

**Wait, why do we need keyword search if we have vector search?** Great question — see [section 4](#4-why-hybrid-search-vector--bm25) below.

**How BM25 works (simplified):**

BM25 is the modern version of **TF-IDF** (Term Frequency — Inverse Document Frequency). It scores chunks based on:

1. **How often your search words appear in the chunk (TF)** — more = higher score
2. **How rare those words are across all chunks (IDF)** — rare words matter more
3. **How long the chunk is** — normalizes for length bias

**Example:** If you search "Massachusetts withholding" in a folder of 10 tax documents:
- Chunks containing both "Massachusetts" AND "withholding" get high scores
- Chunks containing neither get zero
- If "withholding" is rare (only in 2 docs), that word matters more than "Massachusetts" (in 8 docs)

**Implementation:**
```python
# keyword_search.py:20-35 (chunks are dicts; we index their "text" field)
def build(self, chunks):
    self.chunks = chunks
    self.tokenized = [c["text"].lower().split() for c in chunks]
    self.bm25 = BM25Okapi(self.tokenized)

def query(self, query_text, top_k):
    q_tokens = query_text.lower().split()
    scores = self.bm25.get_scores(q_tokens)
    # Sort by score descending, take top_k
```

**Persistence:** The index is saved to `data/bm25_index.pkl` using Python's `pickle`.

**Why this way:** BM25 is fast, needs no GPU, no model downloads, and catches exact-term matches that vector search might miss.

---

### rag_engine.py — Fusing Two Worlds Together

**What it does:** Contains two pure functions: `reciprocal_rank_fusion()` and `format_context()`.

#### Reciprocal Rank Fusion (RRF)

This is how we combine results from vector search and keyword search into a single ranked list.

**The problem:** Vector search gives you 8 results, keyword search gives you 8 results. Some overlap, some don't. Which ones are best?

**The RRF formula:**

```
score(chunk) = 1/(60 + rank_in_vector) + 1/(60 + rank_in_keyword)
```

**Example:**
```
Vector results:   chunk A (rank 1), chunk B (rank 2), chunk C (rank 3)
Keyword results:  chunk C (rank 1), chunk D (rank 2)

RRF scores:
  chunk A: 1/(60+1) = 0.0164
  chunk B: 1/(60+2) = 0.0161
  chunk C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323  ← highest! appears in both
  chunk D: 1/(60+2) = 0.0161
```

Chunks that appear in **both** result lists (like chunk C) get a big boost because they get two scores summed together.

#### format_context()

Takes the top chunks + structured data and builds a plain-text block for the LLM:

```
[STRUCTURED_DATA]
{"w2_wages": 50000, "federal_tax_withheld": 6200, ...}

[SOURCE: w2_sample.pdf page 1]
Box 1: Wages, tips, other compensation..... 50000.00
Box 2: Federal income tax withheld......... 6200.00
```

**Why RRF instead of a fancier method?** RRF is dead simple — no machine learning, no training, no weights to tune. The constant `k=60` is standard in the literature. It works well across very different retrieval methods without calibration.

---

### query_pipeline.py — The Brain That Answers Questions

**What it does:** The `QueryEngine` class orchestrates the full Q&A flow.

**The `_build_prompt` method (the heart of the system):**

```python
def _build_prompt(self, question, chat_history):
    # 1. SEMANTIC SEARCH — embed question, find similar chunks
    vector_hits = self.vector_store.similarity_search_with_score(question, k=8)

    # 2. KEYWORD SEARCH — exact word matching
    keyword_hits = self.keyword_index.query(question, top_k=8)

    # 3. FUSE both lists together
    fused = reciprocal_rank_fusion(vector_hits, keyword_hits)
    top_chunks = fused[:10]  # keep the 10 best

    # 4. LOAD STRUCTURED DATA + optionally calculate metrics
    structured = load_tax_summary(...)
    calc = calculate_metrics(structured) if question looks like math else None

    # 5. FORMAT everything into one context string
    context_text = format_context(top_chunks, structured_payload)

    # 6. BUILD THE PROMPT
    prompt = SYSTEM_PROMPT + history + context + question
    return prompt, context_text, calc, top_chunks
```

**The system prompt:**

```
You are a tax analysis assistant for personal tax return documents.
Use only the provided context and structured data.
Rules:
- Never invent values.
- If data is missing, say what is missing.
- Include source citations.
- Keep answers concise and explicit.
```

**Conversation memory:** The last 5 turns of conversation are included in the prompt so the LLM can handle follow-up questions like "and what about state tax?" without losing context.

**Streaming answer:**

```python
def stream_answer(self, question, chat_history):
    prompt, context_text, calc, top_chunks = self._build_prompt(...)
    # First yield: metadata about what was found
    yield {"type": "meta", "sources": [...], "calculated_metrics": ...}
    # Then stream tokens from the LLM
    for token in self.llm.stream(prompt):
        yield {"type": "token", "content": token}
```

**Why this architecture:** Separating `_build_prompt` from streaming means you could also add a non-streaming `ask()` method (which exists) for testing or batch processing. The two-event protocol (`meta` then `token`) lets the UI display sources before the answer even starts.

---

### llm_client.py — Talking to Ollama

**What it does:** Thin wrapper around LangChain's `ChatOllama`.

```python
class LLMClient:
    def __init__(self, config):
        self.client = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.llm_model,       # qwen3.5:9b
            temperature=config.llm_temperature,  # 0.1
        )

    def invoke(self, prompt):    # returns full response string
    def stream(self, prompt):    # yields tokens one at a time
```

**Temperature = 0.1:** Low temperature means the LLM is more deterministic and factual. For tax questions, you want consistent answers, not creative ones.

**Why this way:** LangChain's `ChatOllama` handles the HTTP streaming, message formatting, and token management. Our wrapper makes it swappable — if you wanted to use OpenAI or Anthropic, you'd only change this file.

---

### structured_extractor.py — Reading Tax Forms by Pattern

**What it does:** Uses regular expressions to pull specific numerical values from known tax form fields.

**Why both RAG AND regex extraction?** See [section 5](#5-why-structured-extraction-on-top-of-rag).

**How it works:**

Each form type has patterns for its key fields:

```python
# W-2 patterns
"w2_wages":  [r"box\s*1\b[^\n$]*wages[^\n$]*(\$?[0-9,]+\.?\d*)"]

# 1040 patterns
"total_income":  [r"(?:line\s*)?9\b[^\n$]*total\s+income[^\n$]*(\$?[0-9,]+\.?\d*)"]

# MA Form-1 patterns
"ma_tax":  [r"massachusetts\s+tax[^\n$]*(\$?[0-9,]+\.?\d*)"]
```

**Audit trail:** Every extraction records:
- Which regex pattern matched
- The exact text snippet that matched
- Character position in the document
- Confidence level

This creates an **auditable** record. You can click "Show Structured Summary" and see not just the values, but exactly where in the document they came from.

**Aggregation:** All extracted values are summed across documents and saved to `tax_summary.json`. If you have two W-2s, `w2_wages` becomes their sum.

**Why this way:** Regex is deterministic and auditable. A human can verify exactly why the system extracted "$50,000" — you can see the matched text. LLMs are probabilistic and might hallucinate. Using both gives you the best of both worlds.

---

### tax_calculator.py — Doing the Math

**What it does:** Computes derived metrics from the structured data.

**Metrics it calculates:**

| Metric | Formula |
|--------|---------|
| Effective federal tax rate | total_tax / adjusted_gross_income × 100 |
| Estimated federal refund | federal_tax_withheld − total_tax |
| Effective MA tax rate | ma_tax / ma_taxable_income × 100 |
| Estimated MA refund | state_tax_withheld − ma_tax |
| Fed vs state difference | total_tax − ma_tax |

**When is this triggered?** Only when the question contains keywords like "effective tax rate," "refund," "difference," "withheld," "how much," or "calculate."

**Why this way:** The LLM is good at language but bad at arithmetic. Rather than asking the LLM to do math (which it might get wrong), we compute metrics precisely in Python and inject the results into the context. The LLM just has to read and report them.

---

## 4. Why Hybrid Search? (Vector + BM25)

This is the most important design decision in the app. Here's why we use both.

### Vector Search (Dense Retrieval)

**Strengths:**
- Understands **meaning and concepts**
- "What was my federal tax rate?" → finds chunks about "federal income tax" even if no single word matches exactly
- Handles synonyms (withholding = tax deducted = FITW)
- Understands context

**Weaknesses:**
- Can miss precise numbers or codes
- "Box 17" → might not find "Box 17" if embeddings focus on nearby dollar amounts
- Needs a model download + GPU-ish performance

### Keyword Search (Sparse Retrieval / BM25)

**Strengths:**
- **Exact match** — "Box 17" finds "Box 17" every time
- Finds rare terms, codes, abbreviations
- Blazing fast (pure math, no neural network)
- Zero setup (just needs tokenized text)

**Weaknesses:**
- Can't handle synonyms or concepts
- "Federal tax" won't match "FITW" unless you ask for it
- Doesn't understand meaning at all

### The sweet spot:

| Question Type | Vector | Keyword |
|--------------|--------|---------|
| "Summarize my total income" | ✅ Great | ❌ Misses |
| "What's in Box 17 of my W-2?" | ❌ Misses | ✅ Finds |
| "Any capital gains?" | ✅ Understands | ❌ Misses |
| "Show me line 35a" | ❌ Misses | ✅ Exact |

**Using both + RRF** gives you the best of both. Chunks that appear in BOTH lists get a huge score boost — those are the most relevant.

### When would you skip one?

- **Vector-only:** If your documents are all about one narrow topic and questions are always conceptual. But you'd miss precise form references.
- **Keyword-only:** If you're searching code or identifiers (like "Find error code E-1042"). But you'd miss conceptual questions.
- **Hybrid (this app):** Tax documents have BOTH conceptual questions ("what's my tax situation?") AND exact references ("line 15, Box 17"). Hybrid handles both.

---

## 5. Why Structured Extraction on Top of RAG?

### The problem with RAG-only for tax forms:

1. **LLMs can hallucinate numbers.** If the LLM sees "wages: 50000" and "federal tax: 6200" in nearby chunks, it might invent a connection.

2. **LLMs are bad at math.** "What's my effective tax rate?" requires dividing two numbers. LLMs get this wrong surprisingly often.

3. **LLMs can't sum across documents.** "What's my total W-2 wages?" with two W-2s requires adding $50,000 + $75,000. LLMs might add wrong or miss a document.

4. **No audit trail.** If the LLM says "$52,300," how do you verify it?

### How structured extraction fixes this:

```
                      ┌─────────────────────────────┐
                      │   W-2 sample.pdf             │
                      │   ─────────────────          │
                      │   Box 1: Wages.....50,000    │
                      │   Box 2: Fed tax....6,200    │
                      │   Box 17: State tax..3,100   │
                      └─────────────────────────────┘
                                    │
                                    ▼
                       regex patterns match
                         (structured_extractor.py)
                                    │
                                    ▼
              ┌──────────────────────────────────────┐
              │  tax_summary.json (on disk)          │
              │  {                                   │
              │    "summary": {                      │
              │      "w2_wages": 50000,              │ ← summed raw fields,
              │      "federal_tax_withheld": 6200,   │   auditable & exact
              │      "state_tax_withheld": 3100      │
              │    },                                │
              │    "summary_audit": { ... }          │ ← where each value came from
              │  }                                   │
              └──────────────────────────────────────┘
                                    │
                  at QUERY TIME (only if the question
                  looks like math), tax_calculator.py
                  computes derived metrics on the fly:
                                    │
                                    ▼
              ┌──────────────────────────────────────┐
              │  calculated_metrics (in memory only) │
              │    "effective_federal_tax_rate": 12.4│ ← computed, never stored
              │    "estimated_federal_refund": ...   │
              └──────────────────────────────────────┘
                                    │
                                    ▼
                        injected into RAG context
                                    │
                                    ▼
                      LLM reads the numbers directly
                      and just reports them.
                      No math, no hallucination.
```

> Note: `tax_summary.json` stores only the **summed raw fields** (under the `summary` key) plus an audit trail. Derived numbers like the effective tax rate are **not** saved — they are recalculated by `tax_calculator.py` each time a calculation-style question is asked.

### The two paths coexist:

| Question Type | Handled By |
|--------------|-----------|
| "What's in Box 1 of my W-2?" | Structured extraction (exact match) |
| "Why might my refund be lower this year?" | RAG / LLM reasoning |
| "Calculate my effective tax rate" | tax_calculator.py (precise math) |
| "What documents did I file?" | LLM reads retrieved chunk metadata |

---

## 6. How a Question Flows Through the System

Let's trace a real example:

**User asks:** "What was my federal tax withholding?"

```
Step 1: chat_action() in app.py
  └─ Checks vector_store has data
  └─ Calls query_engine.stream_answer("What was my federal tax withholding?", [])
```

```
Step 2: _build_prompt() in query_pipeline.py
  └─ VECTOR SEARCH: embeds question, searches ChromaDB → 8 chunks
     Example matches:
       "Box 2: Federal income tax withheld..... 6200.00"  (score: 0.89)
       "Federal withholding summary..."                    (score: 0.76)
       "State tax information..."                          (score: 0.45)

  └─ KEYWORD SEARCH: tokenizes question, BM25 scores → 8 chunks
     Example matches:
       "federal income tax withheld" — exact word match!   (score: 12.4)
       "withholding" — one word match                      (score: 3.2)
       "federal tax" — partial match                       (score: 2.1)

  └─ RRF FUSION: combines both lists → 10 best chunks
     The chunk "Box 2: Federal income tax withheld...6200"
     appears in BOTH lists → gets a big boost → #1 result

  └─ LOAD STRUCTURED DATA:
     tax_summary.json → federal_tax_withheld: 6200

  └─ CHECK IF CALCULATION NEEDED:
     "What was my ..." — doesn't match calc keywords → skip

  └─ FORMAT CONTEXT:
     [STRUCTURED_DATA]
     {"federal_tax_withheld": 6200, ...}
     
     [SOURCE: w2_sample.pdf page 1]
     Box 2: Federal income tax withheld..... 6200.00

  └─ BUILD PROMPT:
     "You are a tax analysis assistant...
      Context:
      [STRUCTURED_DATA]...
      [SOURCE: w2_sample.pdf page 1]...
      Question: What was my federal tax withholding?
      Answer with citations."
```

```
Step 3: stream_answer() yields events
  Event 1: {"type": "meta", "sources": ["w2_sample.pdf"], ...}
  Event 2: {"type": "token", "content": "Your"}
  Event 3: {"type": "token", "content": " federal"}
  Event 4: {"type": "token", "content": " tax"}
  Event 5: {"type": "token", "content": " withholding"}
  Event 6: {"type": "token", "content": " was"}
  Event 7: {"type": "token", "content": " $6,200."}
  ...
```

```
Step 4: app.py receives events
  └─ Meta event → saves source filenames
  └─ Each token event → appends to chatbot message → Gradio re-renders
  └─ After streaming → appends "Source: w2_sample.pdf page 1"
```

**Final output in the chat:**

> **User:** What was my federal tax withholding?
>
> **Assistant:** Your federal tax withholding was $6,200.
>
> Source: w2_sample.pdf page 1

---

## 7. Glossary of RAG Terms

| Term | What It Means | Analogy |
|------|---------------|---------|
| **RAG** | Retrieval-Augmented Generation. Find relevant docs first, then ask the LLM to answer using them. | Open-book exam vs closed-book exam |
| **Embedding** | A list of numbers representing the "meaning" of text. Similar texts have similar numbers. | A GPS coordinate for meaning |
| **Vector Database** | A database that stores embeddings and finds similar ones by "distance" | A library where books are arranged by topic, not alphabetically |
| **Chunk** | A small piece of a document (~800 tokens) | A paragraph rather than a whole chapter |
| **Chunk Overlap** | Overlapping content between neighboring chunks so context isn't cut off | A fade transition between scenes |
| **Token** | A unit of text (~¾ of a word in English). LLMs read tokens, not characters. | Individual LEGO bricks |
| **BM25** | A keyword-matching algorithm that scores documents by how many query words they contain, weighted by rarity | Ctrl+F but smarter about ranking results |
| **RRF** | Reciprocal Rank Fusion — a way to combine two ranked lists into one | Taking two friends' restaurant recommendations and picking the ones both agreed on |
| **Cosine Similarity** | A measure of how similar two vectors are (1.0 = identical, 0.0 = unrelated) | How close two arrows point in the same direction |
| **Semantic Search** | Search by meaning rather than exact words | "Best dog breed for apartments" → finds "top small-space canine companions" |
| **Context Window** | How many tokens an LLM can "see" at once when generating a response | The LLM's working memory |
| **Temperature** | Controls randomness in LLM output (0 = deterministic, 1 = creative) | How much the LLM "goes off-script" |
| **Hybrid Search** | Using both vector search (meaning) AND keyword search (exact match) | Searching by both topic AND author name |

---

## 8. Key Design Decisions Summary

| Decision | Why |
|----------|-----|
| **Local-only (Ollama)** | Privacy. Tax documents are sensitive — never send them to an API. Also free, no API keys. |
| **Hybrid search (vector + BM25)** | Tax forms have both conceptual questions ("summarize income") and exact-reference questions ("Box 17"). Hybrid handles both. |
| **RRF fusion** | Simple, proven, no calibration needed. Works well without training data. |
| **Structured extraction on top of RAG** | Numbers are too important to trust to an LLM. Regex is deterministic, auditable, and never hallucinates. |
| **LangChain wrappers** | Not using LangChain chains/agents (they're complex). Just using it for the thin Ollama + ChromaDB integrations. |
| **Streaming** | Users see the answer build token-by-token, which feels much faster than waiting for the full response. |
| **Folder signature (SHA-256)** | Avoids re-processing unchanged documents on restart. Makes the "it just works" experience smoother. |
| **Knowledge docs as ingested content** | Adding new tax rules means just writing a markdown file in `knowledge/`. No code changes needed. |
| **Pydantic config** | One source of truth for all settings. Overridable via environment variables for deployment flexibility. |
| **Low temperature (0.1)** | Tax answers should be consistent and factual, not creative. |
