from __future__ import annotations

import re
from typing import Any

from config import AppConfig
from keyword_search import KeywordIndex
from llm_client import LLMClient
from rag_engine import format_context, reciprocal_rank_fusion
from structured_extractor import load_tax_summary
from tax_calculator import calculate_metrics
from vector_store import VectorStore


# ── System prompt (instructions the LLM receives before anything else) ──
# This is critical: it constrains the LLM to ONLY use provided context,
# never hallucinate numbers, cite sources, and be concise.
# Without this, the LLM might answer from its general training knowledge
# (which doesn't include the user's personal tax documents).

SYSTEM_PROMPT = """You are a tax analysis assistant for personal tax return documents.
Use only the provided context and structured data.

Rules:
- Never invent values.
- If data is missing, say what is missing.
- Include source citations for factual claims in this format:
  Source: <filename> page <n>
- Keep answers concise and explicit.
"""


def _needs_calculation(question: str) -> bool:
    """
    Heuristic check: does the question ask for a derived metric?

    If the question contains any of these keywords, the pipeline will
    run tax_calculator.calculate_metrics() to compute derived values
    like effective tax rate, refund estimates, etc.

    This is a simple keyword check rather than semantic classification
    because the list of "calculation question" patterns is small and
    enumerable. Calc results are injected into the context so the LLM
    just reports them — no LLM arithmetic needed.
    """
    q = question.lower()
    keywords = ["effective tax rate", "refund", "difference", "withheld", "how much", "calculate"]
    return any(k in q for k in keywords)


def _extract_sources(text: str) -> list[str]:
    """
    Parse [SOURCE: ...] markers out of the context text.

    The context formatter (rag_engine.format_context) embeds source
    citations like [SOURCE: w2_sample.pdf page 1]. This function
    extracts them so they can be displayed to the user separately
    from the generated answer.
    """
    matches = re.findall(r"\[SOURCE:\s*(.*?)\]", text)
    return sorted(set(matches))


class QueryEngine:
    """
    The central orchestrator for the Q&A pipeline.

    Responsibilities:
      1. Hybrid retrieval: vector (semantic) + keyword (BM25) search.
      2. Reciprocal Rank Fusion to merge both result lists.
      3. Structured data loading + optional tax calculation.
      4. Prompt assembly (system + history + context + question).
      5. LLM invocation (streaming or batch).

    Exposes two entry points:
      - ask()           → returns the full answer at once.
      - stream_answer() → yields events for incremental UI updates.
    """

    def __init__(self, config: AppConfig, vector_store: VectorStore, keyword_index: KeywordIndex) -> None:
        self.config = config
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.llm = LLMClient(config)

    def ask(self, question: str, chat_history: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        """
        Non-streaming: build prompt, invoke LLM, return full answer + metadata.

        Useful for testing or batch processing where streaming is not needed.
        """
        prompt, context_text, calc, top_chunks = self._build_prompt(question, chat_history)
        answer = self.llm.invoke(prompt)
        sources = _extract_sources(context_text)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": top_chunks,
            "calculated_metrics": calc,
        }

    def stream_answer(self, question: str, chat_history: list[tuple[str, str]] | None = None):
        """
        Streaming: yields events (dicts) that the UI consumes.

        Event types:
          {"type": "meta", "sources": [...], "retrieved_chunks": [...], "calculated_metrics": ...}
              → yields once at the start, before any tokens.

          {"type": "token", "content": "..."}
              → yields for each token from the LLM.

        This protocol allows the UI to display source citations even
        before the answer begins streaming, and to update incrementally.
        """
        prompt, context_text, calc, top_chunks = self._build_prompt(question, chat_history)
        sources = _extract_sources(context_text)
        yield {"type": "meta", "sources": sources, "retrieved_chunks": top_chunks, "calculated_metrics": calc}
        for token in self.llm.stream(prompt):
            yield {"type": "token", "content": token}

    def _build_prompt(self, question: str, chat_history: list[tuple[str, str]] | None = None):
        """
        Build the complete prompt sent to the LLM.

        This is where all the RAG magic happens:

        1. SEMANTIC SEARCH (Vector)
           - Embed the question using Ollama (nomic-embed-text).
           - Find top-K nearest neighbors in ChromaDB by cosine similarity.
           - Returns documents + distance scores.

        2. KEYWORD SEARCH (BM25)
           - Tokenize the question by simple whitespace split.
           - Score all chunks with BM25 (term frequency + inverse document frequency).
           - Return top-K results with BM25 scores.

        3. FUSION (RRF)
           - Combine both ranked lists using Reciprocal Rank Fusion.
           - Chunks appearing in BOTH lists get boosted scores.
           - Keep the final_top_K best (= 10 by default).

        4. STRUCTURED DATA
           - Load tax_summary.json (aggregated regex extraction results).
           - If the question looks calculation-related, run tax_calculator.
           - Package as structured_payload dict.

        5. CONTEXT FORMATTING
           - Format chunks as [SOURCE: filename page N] blocks.
           - Prepend [STRUCTURED_DATA] section if available.

        6. PROMPT ASSEMBLY
           - System prompt (role + rules).
           - Conversation history (last N turns, for follow-up context).
           - Context block (retrieved chunks + structured data).
           - User question.
           - Answer instruction ("Answer with citations.").

        Returns (prompt_string, context_text, calc_result, top_chunks_list).
        """
        # ── Step 1: Vector (semantic) search ───────────────────────────
        vector_hits_raw = self.vector_store.similarity_search_with_score(question, k=self.config.vector_top_k)
        vector_hits = [
            {"text": doc.page_content, "metadata": doc.metadata, "score": float(score)}
            for doc, score in vector_hits_raw
        ]

        # ── Step 2: Keyword (BM25) search ─────────────────────────────
        keyword_hits = self.keyword_index.query(question, top_k=self.config.keyword_top_k)

        # ── Step 3: Fuse both lists ────────────────────────────────────
        fused = reciprocal_rank_fusion(vector_hits, keyword_hits)
        top_chunks = fused[: self.config.final_top_k]

        # ── Step 4: Load structured data + optional calculation ────────
        structured = load_tax_summary(self.config.structured_dir)
        calc = calculate_metrics(structured) if _needs_calculation(question) else None

        structured_payload = {
            "summary": structured.get("summary", {}),
            "summary_audit": structured.get("summary_audit", {}),
        }
        if calc:
            structured_payload["calculated_metrics"] = calc

        # ── Step 5: Context formatting ─────────────────────────────────
        context_text = format_context(top_chunks, structured_payload)

        # ── Step 6: Prompt assembly ────────────────────────────────────
        prompt_parts = [SYSTEM_PROMPT]

        # Include conversation history for follow-up context (last N turns)
        if chat_history:
            recent = chat_history[-self.config.memory_turns :]
            history_lines = []
            for user_msg, bot_msg in recent:
                history_lines.append(f"User: {user_msg}")
                history_lines.append(f"Assistant: {bot_msg}")
            prompt_parts.append("Conversation:\n" + "\n".join(history_lines))

        prompt_parts.append(f"Context:\n{context_text}")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer with citations.")

        prompt = "\n\n".join(prompt_parts)
        return prompt, context_text, calc, top_chunks
