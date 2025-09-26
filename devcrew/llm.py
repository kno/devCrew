"""Local LLM helpers and wrappers."""
from __future__ import annotations

from typing import Any, Dict, List

from crewai.llm import LLM


class SafeLLM(LLM):
    """`crewai.LLM` variant that guards against trailing assistant messages.

    Some OpenAI-compatible providers reject payloads whose message list ends
    with two consecutive ``assistant`` roles.  crewAI already patches a few
    known providers (e.g. Mistral and Ollama), but self-hosted gateways often
    report ``Cannot have 2 or more assistant messages at the end of the list``
    without advertising a dedicated model prefix.  This subclass mirrors
    crewAI's own mitigation by appending an empty ``user`` message whenever the
    formatted payload would otherwise finish with an assistant turn.  Official
    OpenAI endpoints continue using the default formatting.
    """

    def _format_messages_for_provider(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        formatted = super()._format_messages_for_provider(messages)

        if not formatted or formatted[-1].get("role") != "assistant":
            return formatted

        base_url = getattr(self, "base_url", None) or ""
        if "api.openai.com" in base_url:
            return formatted

        # Avoid mutating the list returned by ``super`` when it happens to be the
        # same instance that crewAI keeps internally.
        patched = list(formatted)
        patched.append({"role": "user", "content": ""})
        return patched


__all__ = ["SafeLLM"]
