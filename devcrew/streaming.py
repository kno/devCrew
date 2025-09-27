"""Streaming helpers to annotate console output with agent information."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, TextIO, Tuple

from crewai.events import (
    AgentExecutionStartedEvent,
    BaseEventListener,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
    TaskStartedEvent,
)
from crewai.events.types.llm_events import LLMCallType


class PrefixedStreamWriter:
    """Write streaming chunks preceded by a single descriptive prefix."""

    def __init__(self, prefix: str, *, stream: Optional[TextIO] = None) -> None:
        self._prefix = prefix
        self._stream = stream or sys.stdout
        self._prefix_printed = False
        self._last_char = "\n"

    def write(self, text: str) -> None:
        if text is None:
            return
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return
        if not self._prefix_printed:
            if self._last_char != "\n":
                self._stream.write("\n")
            self._stream.write(self._prefix)
            self._prefix_printed = True
            if self._prefix:
                self._last_char = self._prefix[-1]
        self._stream.write(text)
        self._stream.flush()
        self._last_char = text[-1]

    def ensure_newline(self) -> None:
        if self._last_char != "\n":
            self._stream.write("\n")
            self._stream.flush()
            self._last_char = "\n"


@dataclass
class _AgentInfo:
    label: str


@dataclass
class _TaskInfo:
    label: str


class ConsoleStreamingPrinter(BaseEventListener):
    """Console listener that annotates streamed tokens with their source."""

    def __init__(self, *, stream: Optional[TextIO] = None) -> None:
        self._stream = stream or sys.stdout
        self._agents: Dict[str, _AgentInfo] = {}
        self._tasks: Dict[str, _TaskInfo] = {}
        self._writers: Dict[Tuple[Any, Any], PrefixedStreamWriter] = {}
        super().__init__()

    # -- BaseEventListener interface -------------------------------------------------
    def setup_listeners(self, crewai_event_bus) -> None:  # type: ignore[override]
        crewai_event_bus.register_handler(TaskStartedEvent, self._on_task_started)
        crewai_event_bus.register_handler(
            AgentExecutionStartedEvent, self._on_agent_execution_started
        )
        crewai_event_bus.register_handler(LLMCallStartedEvent, self._on_llm_call_started)
        crewai_event_bus.register_handler(
            LLMStreamChunkEvent, self._on_llm_stream_chunk
        )
        crewai_event_bus.register_handler(
            LLMCallCompletedEvent, self._on_llm_call_finished
        )
        crewai_event_bus.register_handler(LLMCallFailedEvent, self._on_llm_call_failed)

    # -- Event handlers --------------------------------------------------------------
    def _on_task_started(self, _source: Any, event: TaskStartedEvent) -> None:
        task = getattr(event, "task", None)
        task_id = getattr(task, "id", None)
        if not task_id:
            return
        name = getattr(task, "name", None) or getattr(task, "description", None)
        if not name:
            name = "Tarea"
        self._tasks[str(task_id)] = _TaskInfo(label=str(name))

    def _on_agent_execution_started(
        self, _source: Any, event: AgentExecutionStartedEvent
    ) -> None:
        agent = getattr(event, "agent", None)
        if not agent:
            return
        agent_id = getattr(agent, "id", None)
        if not agent_id:
            return
        name = getattr(agent, "name", None)
        role = getattr(agent, "role", None)
        if name and role and name != role:
            label = f"{name} — {role}"
        else:
            label = name or role or str(agent_id)
        self._agents[str(agent_id)] = _AgentInfo(label=label)

        task = getattr(event, "task", None)
        task_id = getattr(task, "id", None)
        if task_id:
            task_name = getattr(task, "name", None) or getattr(
                task, "description", None
            )
            if task_name:
                self._tasks[str(task_id)] = _TaskInfo(label=str(task_name))

    def _on_llm_call_started(self, _source: Any, event: LLMCallStartedEvent) -> None:
        key = self._event_key(event)
        prefix = self._format_prefix(event)
        self._writers[key] = PrefixedStreamWriter(prefix, stream=self._stream)

    def _on_llm_stream_chunk(self, _source: Any, event: LLMStreamChunkEvent) -> None:
        key = self._event_key(event)
        writer = self._writers.get(key)
        if writer is None:
            prefix = self._format_prefix(event)
            writer = PrefixedStreamWriter(prefix, stream=self._stream)
            self._writers[key] = writer
        chunk = getattr(event, "chunk", "")
        if chunk:
            writer.write(chunk)

    def _on_llm_call_finished(self, _source: Any, event: LLMCallCompletedEvent) -> None:
        key = self._event_key(event)
        writer = self._writers.pop(key, None)
        if writer:
            writer.ensure_newline()

    def _on_llm_call_failed(self, _source: Any, event: LLMCallFailedEvent) -> None:
        key = self._event_key(event)
        writer = self._writers.pop(key, None)
        prefix = self._format_prefix(event)
        if writer is None:
            writer = PrefixedStreamWriter(prefix, stream=self._stream)
        error = getattr(event, "error", "Error desconocido")
        writer.write(f"[Error: {error}]\n")
        writer.ensure_newline()

    # -- Helpers ---------------------------------------------------------------------
    @staticmethod
    def _event_key(event: Any) -> Tuple[Any, Any]:
        return (getattr(event, "agent_id", None), getattr(event, "task_id", None))

    def _format_prefix(self, event: Any) -> str:
        agent_label = self._agents.get(str(getattr(event, "agent_id", "")))
        task_label = self._tasks.get(str(getattr(event, "task_id", "")))
        call_type = getattr(event, "call_type", None)

        header: str
        if call_type == LLMCallType.TOOL_CALL:
            tool_call = getattr(event, "tool_call", None)
            tool_name: Optional[str] = None
            if tool_call is not None:
                tool_name = getattr(tool_call, "name", None)
                if not tool_name:
                    function = getattr(tool_call, "function", None)
                    if isinstance(function, dict):
                        tool_name = function.get("name")
                    else:
                        tool_name = getattr(function, "name", None)
            header = "Herramienta"
            if tool_name:
                header = f"Herramienta {tool_name}"
        elif agent_label:
            header = f"Agente {agent_label.label}"
        elif getattr(event, "agent_role", None):
            header = f"Agente {getattr(event, 'agent_role')}"
        else:
            header = "Orquestador"

        suffix_parts = []
        if task_label:
            suffix_parts.append(f"Tarea {task_label.label}")
        model = getattr(event, "model", None)
        if model:
            suffix_parts.append(str(model))

        if suffix_parts:
            header = f"{header} · {' · '.join(suffix_parts)}"

        return f"▶ {header}:\n"


__all__ = ["ConsoleStreamingPrinter", "PrefixedStreamWriter"]
