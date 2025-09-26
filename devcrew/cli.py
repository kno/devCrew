"""Command line interface for the dynamic crew orchestrator."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from crewai import LLM

from .orchestrator import (
    AgentPlan,
    CrewPlan,
    DynamicCrewOrchestrator,
    PlanGenerationError,
)
from .tools import build_default_tool_registry


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    else:
        root.setLevel(level)


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return args.prompt_file.read()
    return sys.stdin.read()


def _create_chat_model(*, model: str, temperature: float, provider: Optional[str]) -> BaseChatModel:
    if provider:
        try:
            return init_chat_model(model=model, model_provider=provider, temperature=temperature, streaming=True)
        except Exception as exc:
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
    # --- Planner ---
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

    # --- Agents ---
    agent_provider = _resolve_provider(
        args.agent_provider,
        "AGENT_MODEL_PROVIDER",
        "MODEL_PROVIDER",
        "LLM_PROVIDER",
    ) or "openai"

    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "http://192.168.68.118:8080/v1"
    )
    api_key = os.getenv("OPENAI_API_KEY", "sk-no-key-needed")

    def agent_factory(agent_plan: AgentPlan):
        model_name = args.agent_model or args.planner_model
        model_id = f"{agent_provider}/{model_name}"
        return LLM(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=args.agent_temperature,
            stream=args.stream,  # <- habilita streaming en agentes
        )

    registry = build_default_tool_registry()
    return DynamicCrewOrchestrator(
        planner_llm=planner_llm,
        agent_llm_factory=agent_factory,
        tool_registry=registry,
        verbose=args.verbose,
    )


def _prepare_output_dir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_outputs(outdir: Path, plan, result, fmt: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = outdir / f"plan_{timestamp}.json"
    result_file = outdir / f"result_{timestamp}.{ 'json' if fmt=='json' else 'txt'}"

    with plan_file.open("w", encoding="utf-8") as fh:
        json.dump(plan.model_dump(), fh, indent=2, ensure_ascii=False)

    if fmt == "json":
        out = {
            "summary": plan.summary,
            "process": plan.process,
            "agents": [a.name for a in plan.agents],
            "tasks": [t.name for t in plan.tasks],
            "result": result,
        }
        with result_file.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
    else:
        with result_file.open("w", encoding="utf-8") as fh:
            if isinstance(result, (dict, list)):
                fh.write(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                fh.write(str(result))

    logging.info("Outputs saved: %s , %s", plan_file, result_file)


def _prompt_for_execution(plan: CrewPlan) -> bool:
    """Ask the user whether the freshly built crew should be executed."""

    prompt = (
        f"Crew planned with {len(plan.agents)} agents and {len(plan.tasks)} tasks. "
        "Run it now? [y/N]: "
    )
    try:
        response = input(prompt)
    except EOFError:
        logging.info("No input available to confirm execution; skipping run.")
        return False

    return response.strip().lower() in {"y", "yes"}


async def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()

    default_planner_model = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
    default_agent_model = os.getenv("AGENT_MODEL")
    default_planner_provider = _resolve_provider(None, "PLANNER_MODEL_PROVIDER", "MODEL_PROVIDER", "LLM_PROVIDER")
    default_agent_provider = _resolve_provider(None, "AGENT_MODEL_PROVIDER", "MODEL_PROVIDER", "LLM_PROVIDER")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Problem statement to solve. If omitted, read from stdin.")
    parser.add_argument("-f", "--prompt-file", type=argparse.FileType("r"), help="Read the prompt from a file.")
    parser.add_argument("--planner-model", default=default_planner_model, help="Model used by the planning agent.")
    parser.add_argument("--planner-provider", default=default_planner_provider, help="Provider for the planner model.")
    parser.add_argument("--agent-model", default=default_agent_model, help="Model used by execution agents.")
    parser.add_argument("--agent-provider", default=default_agent_provider, help="Provider for execution agent models.")
    parser.add_argument("--planner-temperature", type=float, default=0.1, help="Temperature for the planner model.")
    parser.add_argument("--agent-temperature", type=float, default=0.3, help="Temperature for execution agents.")
    parser.add_argument("--dry-run", action="store_true", help="Only show the generated plan without executing the crew.")
    parser.add_argument("--show-plan", action="store_true", help="Print the plan in JSON format.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Automatically run the crew after planning without prompting.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory to save plan and results (default: ./outputs)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format for the final result.")
    parser.add_argument("--stream", action="store_true", help="Stream token-by-token from agent LLMs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging and verbose agents.")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        # Registrar listener de streaming si procede
        if args.stream and BaseEventListener is not object:
            listener = TokenStreamListener()
            if _get_bus:
                bus = _get_bus()
                # Algunas versiones requieren .add_listener, otras instanciar el listener basta
                try:
                    bus.add_listener(listener)  # type: ignore[attr-defined]
                except Exception:
                    # Si el bus inyecta listeners automáticamente vía entrypoints, ignoramos
                    pass

        orchestrator = build_orchestrator(args)
        prompt = _load_prompt(args)
        outdir = _prepare_output_dir(args.output_dir)

        if args.dry_run:
            plan = orchestrator.plan(prompt)
            if args.show_plan:
                print(json.dumps(plan.model_dump(), indent=2))
            _save_outputs(outdir, plan, {}, "json")
            return 0

        built = orchestrator.plan_and_build(prompt)
        plan = built.plan

        if args.show_plan:
            print("\n=== Plan (JSON) ===\n")
            print(json.dumps(plan.model_dump(), indent=2))

        if args.execute:
            should_run = True
        else:
            should_run = _prompt_for_execution(plan)

        executed = False
        final_result = None
        saved_result: Any

        if should_run:
            run_out = orchestrator.kickoff(built, prompt)
            final_result = run_out.get("result")
            executed = True
            saved_result = final_result
        else:
            skip_message = "Execution skipped by user request."
            logging.info(skip_message)
            saved_result = {"status": "skipped", "message": skip_message}

        if executed:
            print("\n=== Crew output ===\n")
            if args.format == "json":
                out = {
                    "summary": plan.summary,
                    "process": plan.process,
                    "agents": [a.name for a in plan.agents],
                    "tasks": [t.name for t in plan.tasks],
                    "result": saved_result,
                }
                print(json.dumps(out, indent=2, ensure_ascii=False))
            else:
                if isinstance(final_result, (dict, list)):
                    print(json.dumps(final_result, indent=2, ensure_ascii=False))
                else:
                    print(str(final_result))
        else:
            print("\nCrew execution skipped.\n")
            if args.format == "json":
                out = {
                    "summary": plan.summary,
                    "process": plan.process,
                    "agents": [a.name for a in plan.agents],
                    "tasks": [t.name for t in plan.tasks],
                    "result": saved_result,
                }
                print(json.dumps(out, indent=2, ensure_ascii=False))
            else:
                print(saved_result["message"])

        _save_outputs(outdir, plan, saved_result, args.format)
        return 0

    except PlanGenerationError as e:
        logging.error("%s", e)
        return 2
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(asyncio.run(main()))
