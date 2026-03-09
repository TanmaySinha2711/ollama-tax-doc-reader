from __future__ import annotations

from collections.abc import Iterator

from langchain_ollama import ChatOllama

from config import AppConfig


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.llm_model,
            temperature=config.llm_temperature,
        )

    def invoke(self, prompt: str) -> str:
        result = self.client.invoke(prompt)
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
        for chunk in self.client.stream(prompt):
            content = getattr(chunk, "content", "")
            if content:
                yield content
