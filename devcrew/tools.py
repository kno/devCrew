"""Utility classes and default tools for the dynamic crew orchestrator."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from crewai.tools import BaseTool
from crewai_tools import FileReadTool, FileWriterTool, ScrapeWebsiteTool, DirectoryReadTool

import os
import requests


class ToolRegistry:
    """Registry that keeps track of available tools by name.

    The orchestrator receives a registry instance to resolve tool names coming
    from the planning step into concrete :class:`~crewai.tools.BaseTool`
    implementations.
    """

    def __init__(self, tools: Optional[Iterable[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if tools is not None:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Parameters
        ----------
        tool:
            Instance of a crewAI tool. The value of :attr:`BaseTool.name` is used
            as the lookup key.
        """

        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry if it exists."""

        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        """Return the tool registered under ``name``.

        Raises
        ------
        KeyError
            If the tool is not present in the registry.
        """

        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check whether a tool exists in the registry."""

        return name in self._tools

    def available(self) -> List[str]:
        """Return the list of registered tool names."""

        return sorted(self._tools)


class CalculatorTool(BaseTool):
    """A small calculator tool that evaluates basic Python expressions."""

    name: str = "calculator"
    description: str = (
        "Perform short mathematical calculations. "
        "The input should be a valid Python arithmetic expression such as "
        "'2 * (3 + 5)' or 'round(3.1415, 2)'."
    )

    def _run(self, query: str) -> str:
        try:
            # ``eval`` is safe here because we restrict the globals/locals and
            # rely only on Python's mathematical operators and functions.
            value = eval(query, {"__builtins__": {}}, {})  # noqa: S307
        except Exception as exc:  # pragma: no cover - defensive programming
            return f"Calculator error: {exc}"
        return str(value)

    async def _arun(self, query: str) -> str:  # pragma: no cover - async path
        return self._run(query)

class SearxngSearchTool(BaseTool):
  """Busca en SearXNG por query y devuelve un resumen."""

  name: str = "searxng_search"
  description: str = "Busca en SearXNG por query y devuelve un resumen."

  def _run(self, query: str) -> str:
    url = os.getenv("SEARX_URL", "http://localhost:8081")
    try:
      r = requests.get(
        f"{url}/search",
        params={"q": query, "format": "json"},
        timeout=20
      )
      r.raise_for_status()
      data = r.json()
      results = [
        f"- {it.get('title','')} | {it.get('url','')} | {it.get('content','')}"
        for it in data.get("results", [])[:10]
      ]
      return "\n".join(results) or "Sin resultados"
    except Exception as exc:
      return f"Error en la búsqueda: {exc}"

  async def _arun(self, query: str) -> str:
    return self._run(query)


def build_default_tool_registry() -> ToolRegistry:
    """Create a :class:`ToolRegistry` pre-populated with common tools."""

    return ToolRegistry(tools=[SearxngSearchTool(), FileReadTool(), FileWriterTool(), ScrapeWebsiteTool(), DirectoryReadTool()])
