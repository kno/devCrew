"""Command line interface for the dynamic crew orchestrator."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .orchestrator import AgentPlan, DynamicCrewOrchestrator
from .tools import build_default_tool_registry


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return args.prompt_file.read()
    return sys.stdin.read()


def _create_chat_model(*, model: str, temperature: float, provider: Optional[str]) -> BaseChatModel:
    """Instantiate a chat model for the given provider.

    Parameters
    ----------
    model:
        Identifier of the target model (e.g. ``gpt-4o-mini``).
    temperature:
        Sampling temperature for the model.
    provider:
        Provider name understood by :func:`langchain.chat_models.init_chat_model`.
        When ``None`` the default OpenAI chat model implementation is used.
    """

    if provider:
        try:
            return init_chat_model(model=model, model_provider=provider, temperature=temperature)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to initialise chat model '{model}' for provider '{provider}'."
            ) from exc
    return ChatOpenAI(model=model, temperature=temperature)


def _resolve_provider(cli_value: Optional[str], *env_vars: str) -> Optional[str]:
    if cli_value:
        return cli_value
    for var in env_vars:
        if value := os.getenv(var):
            return value
    return None


def build_orchestrator(args: argparse.Namespace) -> DynamicCrewOrchestrator:
    planner_provider = _resolve_provider(
        args.planner_provider,
        "PLANNER_MODEL_PROVIDER",
        "MODEL_PROVIDER",
        "LLM_PROVIDER",
    )
    planner_llm = _create_chat_model(
        model=args.planner_model,
        temperature=args.planner_temperature,
        provider=planner_provider,
    )

    agent_provider = _resolve_provider(
        args.agent_provider,
        "AGENT_MODEL_PROVIDER",
        "MODEL_PROVIDER",
        "LLM_PROVIDER",
    )

    def agent_factory(agent_plan: AgentPlan):
        model_name = args.agent_model or args.planner_model
        return _create_chat_model(model=model_name, temperature=args.agent_temperature, provider=agent_provider)

    registry = build_default_tool_registry()
    return DynamicCrewOrchestrator(
        planner_llm=planner_llm,
        agent_llm_factory=agent_factory,
        tool_registry=registry,
        verbose=args.verbose,
    )


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    default_planner_model = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
    default_agent_model = os.getenv("AGENT_MODEL")
    default_planner_provider = _resolve_provider(None, "PLANNER_MODEL_PROVIDER", "MODEL_PROVIDER", "LLM_PROVIDER")
    default_agent_provider = _resolve_provider(None, "AGENT_MODEL_PROVIDER", "MODEL_PROVIDER", "LLM_PROVIDER")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Problem statement to solve. If omitted, read from stdin.")
    parser.add_argument("-f", "--prompt-file", type=argparse.FileType("r"), help="Read the prompt from a file.")
    parser.add_argument("--planner-model", default=default_planner_model, help="Model used by the planning agent.")
    parser.add_argument("--planner-provider", default=default_planner_provider, help="Provider for the planner model (e.g. openai, anthropic, ollama).")
    parser.add_argument("--agent-model", default=default_agent_model, help="Model used by execution agents (defaults to planner model).")
    parser.add_argument("--agent-provider", default=default_agent_provider, help="Provider for execution agent models.")
    parser.add_argument("--planner-temperature", type=float, default=0.1, help="Temperature for the planner model.")
    parser.add_argument("--agent-temperature", type=float, default=0.3, help="Temperature for execution agents.")
    parser.add_argument("--dry-run", action="store_true", help="Only show the generated plan without executing the crew.")
    parser.add_argument("--show-plan", action="store_true", help="Print the plan in JSON format before execution.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging and verbose agents.")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    orchestrator = build_orchestrator(args)
    prompt = _load_prompt(args)

    plan = orchestrator.plan(prompt)
    if args.show_plan or args.dry_run:
        print(json.dumps(plan.model_dump(), indent=2))
    if args.dry_run:
        return 0

    result = orchestrator.build_crew(plan).crew.kickoff(inputs={"problem": prompt})
    print("\n=== Crew output ===\n")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
