"""Command line interface for the dynamic crew orchestrator."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

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


def build_orchestrator(args: argparse.Namespace) -> DynamicCrewOrchestrator:
    planner_llm = ChatOpenAI(model=args.planner_model, temperature=args.planner_temperature)

    def agent_factory(agent_plan: AgentPlan):
        return ChatOpenAI(model=args.agent_model or args.planner_model, temperature=args.agent_temperature)

    registry = build_default_tool_registry()
    return DynamicCrewOrchestrator(
        planner_llm=planner_llm,
        agent_llm_factory=agent_factory,
        tool_registry=registry,
        verbose=args.verbose,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Problem statement to solve. If omitted, read from stdin.")
    parser.add_argument("-f", "--prompt-file", type=argparse.FileType("r"), help="Read the prompt from a file.")
    parser.add_argument("--planner-model", default="gpt-4o-mini", help="Model used by the planning agent.")
    parser.add_argument("--agent-model", default=None, help="Model used by execution agents (defaults to planner model).")
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
