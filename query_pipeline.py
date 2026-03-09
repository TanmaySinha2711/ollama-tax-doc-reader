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
    q = question.lower()
    keywords = ["effective tax rate", "refund", "difference", "withheld", "how much", "calculate"]
    return any(k in q for k in keywords)


def _extract_sources(text: str) -> list[str]:
    matches = re.findall(r"\[SOURCE:\s*(.*?)\]", text)
    return sorted(set(matches))


class QueryEngine:
    def __init__(self, config: AppConfig, vector_store: VectorStore, keyword_index: KeywordIndex) -> None:
        self.config = config
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.llm = LLMClient(config)

    def ask(self, question: str, chat_history: list[tuple[str, str]] | None = None) -> dict[str, Any]:
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
        prompt, context_text, calc, top_chunks = self._build_prompt(question, chat_history)
        sources = _extract_sources(context_text)
        yield {"type": "meta", "sources": sources, "retrieved_chunks": top_chunks, "calculated_metrics": calc}
        for token in self.llm.stream(prompt):
            yield {"type": "token", "content": token}

    def _build_prompt(self, question: str, chat_history: list[tuple[str, str]] | None = None):
        vector_hits_raw = self.vector_store.similarity_search_with_score(question, k=self.config.vector_top_k)
        vector_hits = [
            {"text": doc.page_content, "metadata": doc.metadata, "score": float(score)}
            for doc, score in vector_hits_raw
        ]

        keyword_hits = self.keyword_index.query(question, top_k=self.config.keyword_top_k)

        fused = reciprocal_rank_fusion(vector_hits, keyword_hits)
        top_chunks = fused[: self.config.final_top_k]

        structured = load_tax_summary(self.config.structured_dir)
        calc = calculate_metrics(structured) if _needs_calculation(question) else None

        structured_payload = {
            "summary": structured.get("summary", {}),
            "summary_audit": structured.get("summary_audit", {}),
        }
        if calc:
            structured_payload["calculated_metrics"] = calc

        context_text = format_context(top_chunks, structured_payload)

        prompt_parts = [SYSTEM_PROMPT]
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
