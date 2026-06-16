from __future__ import annotations

from collections.abc import Iterator

from langchain_ollama import ChatOllama

from config import AppConfig


class LLMClient:
    """
    Thin wrapper around LangChain's ChatOllama.

    Handles:
      - Connection to a local Ollama instance.
      - Model selection (qwen3.5:9b by default).
      - Temperature control (0.1 for deterministic tax answers).
      - Both batch (invoke) and streaming (stream) modes.

    The wrapper normalizes LangChain's response format (which varies
    between different result types) into plain strings. If you wanted
    to switch to a different LLM provider (OpenAI, Anthropic, etc.),
    this is the only file you'd need to change.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.llm_model,
            temperature=config.llm_temperature,
        )

    def invoke(self, prompt: str) -> str:
        """
        Send the prompt and wait for the complete response.

        Returns the full response text as a single string.
        Used by QueryEngine.ask() for non-streaming scenarios.
        """
        result = self.client.invoke(prompt)
        # LangChain returns an AIMessage object. Its .content can be:
        # - a string (most common)
        # - a list of content blocks (multimodal, tool calls)
        # We normalize to string.
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return str(content)

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the response token-by-token.

        Yields individual token strings as they arrive from Ollama.
        Each token is the .content attribute of a LangChain chunk object.

        Used by QueryEngine.stream_answer() for the Gradio streaming UI.
        """
        for chunk in self.client.stream(prompt):
            content = getattr(chunk, "content", "")
            if content:
                yield content
