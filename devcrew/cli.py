"""Command line interface for the dynamic crew orchestrator."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .llm import SafeLLM
from .plan_selection import ensure_prompt_for_plan, select_saved_plan
from .plan_storage import (
    dump_plan_on_failure,
    initialise_plan_storage,
    persist_plan_result,
    plan_to_yaml_text,
    prepare_output_dir,
    update_plan_metadata,
)

try:  # Optional streaming dependencies (not present in all crewAI installs)
    from crewai.events.base_event_listener import BaseEventListener
except Exception:  # pragma: no cover - defensive import shim
    try:
        from crewai.utilities.events.base_event_listener import BaseEventListener  # type: ignore
    except Exception:  # pragma: no cover - best-effort fallback
        BaseEventListener = object  # type: ignore[assignment]

try:  # pragma: no cover - optional import path variations
    from crewai.events.event_bus import crewai_event_bus as _get_bus
except Exception:  # pragma: no cover - fallback when event bus is unavailable
    _get_bus = None

TokenStreamListener = None
_token_listener_module = None
for _candidate in (
    "crewai.listeners.token_stream_listener",
    "crewai.events.listeners.streaming.token_stream_listener",
):
    if TokenStreamListener is not None:
        break
    try:  # pragma: no cover - optional dependency discovery
        _token_listener_module = __import__(_candidate, fromlist=["TokenStreamListener"])
        TokenStreamListener = getattr(_token_listener_module, "TokenStreamListener", None)
    except Exception:
        TokenStreamListener = None

del _candidate
del _token_listener_module

from .orchestrator import (
    AgentPlan,
    CrewPlan,
    DynamicCrewOrchestrator,
    PlanGenerationError,
    BuiltCrew,
)
from .tools import build_default_tool_registry


LOGGER = logging.getLogger(__name__)


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
        return SafeLLM(
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


def _prompt_for_execution(plan: CrewPlan) -> bool:
    """Ask the user whether the freshly built crew should run."""

    if not sys.stdin.isatty():
        LOGGER.info("Non-interactive terminal detected; skipping execution.")
        return False

    summary = plan.summary or "the planned crew"
    print(f"\nPlan ready for: {summary}")

    while True:
        try:
            response = input("Execute the crew now? [y/N]: ")
        except EOFError:
            LOGGER.info("No confirmation received; skipping execution.")
            return False

        answer = response.strip().lower()
        if answer in {"", "n", "no"}:
            return False
        if answer in {"y", "yes"}:
            return True

        print("Please answer 'y' or 'n'.")


def _register_stream_listener(args: argparse.Namespace) -> None:
    """Attach the optional crewAI token stream listener if available."""

    if not args.stream:
        return

    listener_cls = TokenStreamListener
    bus_getter = _get_bus if callable(_get_bus) else None

    if listener_cls is None or bus_getter is None:
        LOGGER.debug("Streaming requested but listener hooks are unavailable; skipping.")
        return

    if not isinstance(listener_cls, type):  # pragma: no cover - defensive guard
        LOGGER.debug("TokenStreamListener is not instantiable; skipping stream listener.")
        return

    if BaseEventListener is not object and isinstance(BaseEventListener, type):
        if not issubclass(listener_cls, BaseEventListener):
            LOGGER.debug(
                "TokenStreamListener does not inherit from BaseEventListener; skipping stream listener."
            )
            return

    try:
        bus = bus_getter()
    except Exception:  # pragma: no cover - optional dependency behaviour
        LOGGER.debug("Failed to obtain crewAI event bus; skipping stream listener.", exc_info=True)
        return

    if bus is None:
        LOGGER.debug("crewAI event bus helper returned None; skipping stream listener registration.")
        return

    try:
        listener = listener_cls()
    except Exception:  # pragma: no cover - optional dependency behaviour
        LOGGER.debug("Could not instantiate TokenStreamListener; skipping registration.", exc_info=True)
        return

    add_listener = getattr(bus, "add_listener", None)
    if callable(add_listener):
        try:
            add_listener(listener)
        except Exception:  # pragma: no cover - optional dependency behaviour
            LOGGER.debug(
                "crewAI event bus rejected token stream listener; assuming auto-registration.",
                exc_info=True,
            )


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
    parser.add_argument("--show-plan", action="store_true", help="Print the plan in YAML format.")
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

    prompt: Optional[str] = None
    plan: Optional[CrewPlan] = None
    built: Optional[BuiltCrew] = None
    plan_dir: Optional[Path] = None
    result_persisted = False

    try:
        _register_stream_listener(args)

        orchestrator = build_orchestrator(args)
        outdir = prepare_output_dir(args.output_dir)

        run_existing_plan = args.execute and not args.prompt and not args.prompt_file

        if run_existing_plan:
            if args.dry_run:
                logging.error("--dry-run cannot be combined with executing a saved plan.")
                return 2

            selection, cancelled = select_saved_plan(outdir)
            if not selection:
                return 0 if cancelled else 1

            plan = selection.plan
            prompt = ensure_prompt_for_plan(selection)
            if not prompt:
                return 1

            plan_dir = initialise_plan_storage(
                outdir,
                plan,
                prompt,
                source=selection.path,
            )

            built = orchestrator.build_crew(plan)
            should_run = True
        else:
            prompt = _load_prompt(args)
            plan = orchestrator.plan(prompt)

            if args.dry_run:
                if args.show_plan:
                    print(plan_to_yaml_text(plan))
                plan_dir = initialise_plan_storage(outdir, plan, prompt)
                persist_plan_result(plan_dir, plan, {}, "json", status="dry-run")
                result_persisted = True
                return 0

            plan_dir = initialise_plan_storage(outdir, plan, prompt)
            built = orchestrator.build_crew(plan)
            should_run = args.execute or _prompt_for_execution(plan)

        assert plan is not None and built is not None and prompt is not None and plan_dir is not None

        if args.show_plan:
            print("\n=== Plan (YAML) ===\n")
            print(plan_to_yaml_text(plan))

        executed = False
        final_result = None
        saved_result: Any

        if should_run:
            run_out = orchestrator.kickoff(built, prompt)
            final_result = run_out.get("result")
            executed = True
            saved_result = final_result
            persist_plan_result(plan_dir, plan, saved_result, args.format, status="executed")
            result_persisted = True
        else:
            skip_message = "Execution skipped by user request."
            logging.info(skip_message)
            saved_result = {"status": "skipped", "message": skip_message}
            persist_plan_result(plan_dir, plan, saved_result, args.format, status="skipped")
            result_persisted = True

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

        return 0

    except PlanGenerationError as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        if plan_dir:
            now_iso = datetime.now().isoformat(timespec="seconds")
            update_plan_metadata(
                plan_dir,
                status="cancelled",
                executed=False,
                updated_at=now_iso,
                completed_at=now_iso,
            )
        logging.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - defensive guard
        if plan is not None:
            dump_plan_on_failure(plan, plan_dir)

        if plan_dir and plan is not None and not result_persisted:
            now_iso = datetime.now().isoformat(timespec="seconds")
            update_plan_metadata(
                plan_dir,
                status="error",
                executed=False,
                error=str(exc),
                updated_at=now_iso,
                completed_at=now_iso,
            )
        logging.exception("Unhandled error: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(asyncio.run(main()))
