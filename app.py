from __future__ import annotations

from pathlib import Path

import gradio as gr

from config import get_config
from ingest import ingest_folder
from keyword_search import KeywordIndex
from query_pipeline import QueryEngine
from structured_extractor import load_tax_summary
from vector_store import VectorStore


config = get_config()
vector_store = VectorStore(config)
keyword_index = KeywordIndex()

if config.bm25_path.exists():
    try:
        keyword_index.load(config.bm25_path)
    except Exception:
        keyword_index = KeywordIndex()

query_engine = QueryEngine(config, vector_store, keyword_index)


def ingest_action(folder_path: str, force_reingest: bool) -> str:
    if not folder_path:
        return "Please provide a folder path."
    try:
        result = ingest_folder(
            folder=Path(folder_path),
            config=config,
            vector_store=vector_store,
            keyword_index=keyword_index,
            force=force_reingest,
        )
    except Exception as exc:
        return f"Ingestion failed: {exc}"

    if result["status"] == "skipped":
        return f"Ingestion skipped. {result['reason']}"

    return (
        f"Ingestion completed. PDFs: {result['pdf_count']} | "
        f"Chunks: {result['chunk_count']} | Folder: {result['folder']}"
    )


def chat_action(message: str, history: list[dict[str, str]] | None):
    history = history or []
    if not message.strip():
        yield history, ""
        return

    if vector_store.count() == 0:
        reply = "No ingested data found. Please ingest a tax document folder first."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        yield history, ""
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, ""

    reply_parts: list[str] = []
    source_lines = ""
    for event in query_engine.stream_answer(message, _history_as_pairs(history[:-2])):
        if event.get("type") == "meta":
            sources = event.get("sources", [])
            source_lines = "\n".join(f"Source: {s}" for s in sources[:8])
            continue

        token = event.get("content", "")
        reply_parts.append(token)
        current = "".join(reply_parts)
        history[-1]["content"] = current
        yield history, ""

    final_reply = "".join(reply_parts).strip()
    if source_lines:
        final_reply = f"{final_reply}\n\n{source_lines}" if final_reply else source_lines
    history[-1]["content"] = final_reply
    yield history, ""


def _history_as_pairs(history: list[dict[str, str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    current_user = ""
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            current_user = content
        elif role == "assistant" and current_user:
            pairs.append((current_user, content))
            current_user = ""
    return pairs


def structured_summary_action() -> str:
    summary = load_tax_summary(config.structured_dir)
    s = summary.get("summary", {})
    if not s:
        return "No structured tax summary available yet."
    lines = ["Extracted summary fields:"]
    for k, v in sorted(s.items()):
        lines.append(f"- {k}: {v}")
    audit = summary.get("summary_audit", {})
    if audit:
        lines.append("")
        lines.append("Audit evidence:")
        for field, evidences in sorted(audit.items()):
            if not evidences:
                continue
            e = evidences[0]
            source_doc = e.get("source_document", "unknown")
            snippet = e.get("matched_text", "")
            confidence = e.get("confidence", "unknown")
            lines.append(f"- {field}: {confidence} | {source_doc} | {snippet}")
    return "\n".join(lines)


with gr.Blocks(title="Local Tax AI Assistant") as demo:
    gr.Markdown("## Local Tax AI Assistant (Ollama + Hybrid RAG)")

    with gr.Row():
        folder_input = gr.Textbox(label="Tax document folder path", placeholder="/path/to/tax_docs", scale=4)
        force_checkbox = gr.Checkbox(label="Force re-ingest", value=False, scale=1)
    ingest_btn = gr.Button("Ingest Documents")
    ingest_status = gr.Textbox(label="Ingestion status", interactive=False)

    summary_btn = gr.Button("Show Structured Summary")
    summary_box = gr.Textbox(label="Structured summary", lines=12, interactive=False)

    chatbot = gr.Chatbot(label="Tax Chat")
    user_msg = gr.Textbox(label="Ask a tax question")
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")

    ingest_btn.click(ingest_action, inputs=[folder_input, force_checkbox], outputs=[ingest_status])
    summary_btn.click(structured_summary_action, inputs=None, outputs=[summary_box])
    send_btn.click(chat_action, inputs=[user_msg, chatbot], outputs=[chatbot, user_msg])
    clear_btn.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    demo.queue().launch()
