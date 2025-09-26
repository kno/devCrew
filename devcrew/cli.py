"""Command line interface for the dynamic crew orchestrator."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

import yaml
from crewai import LLM

from .orchestrator import (
    AgentPlan,
    CrewPlan,
    DynamicCrewOrchestrator,
    PlanGenerationError,
    BuiltCrew,
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


def _yaml_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def _yaml_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _plan_to_yaml_payload(plan: CrewPlan) -> dict[str, Any]:
    agents = [agent.model_dump() for agent in plan.agents]
    tasks = [task.model_dump() for task in plan.tasks]
    return {
        "crew": {
            "summary": plan.summary,
            "process": plan.process,
            "agents": [a.get("name") for a in agents],
            "tasks": [t.get("name") for t in tasks],
        },
        "agents": agents,
        "tasks": tasks,
    }


def _plan_to_yaml_text(plan: CrewPlan) -> str:
    return yaml.safe_dump(
        _plan_to_yaml_payload(plan), sort_keys=False, allow_unicode=True
    )


def _write_plan_bundle(plan_dir: Path, plan: CrewPlan, prompt: Optional[str]) -> None:
    bundle = _plan_to_yaml_payload(plan)
    plan_dir.mkdir(parents=True, exist_ok=True)

    _yaml_dump(plan_dir / "plan.yaml", bundle)
    _yaml_dump(plan_dir / "agents.yaml", {"agents": bundle["agents"]})
    _yaml_dump(plan_dir / "tasks.yaml", {"tasks": bundle["tasks"]})
    _yaml_dump(plan_dir / "crew.yaml", bundle["crew"])

    metadata: dict[str, Any] = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if prompt:
        metadata["prompt"] = prompt
    _yaml_dump(plan_dir / "metadata.yaml", metadata)


def _extract_section(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        section = data.get(key, [])
    else:
        section = data

    if section is None:
        return []
    if isinstance(section, list):
        return section

    raise ValueError(f"Unsupported {key} structure in saved plan: {type(section)!r}")


def _read_plan_bundle(plan_dir: Path) -> tuple[CrewPlan, Optional[str]]:
    try:
        plan_data = _yaml_load(plan_dir / "plan.yaml")
    except FileNotFoundError:
        plan_data = None

    if isinstance(plan_data, dict):
        agents_data = _extract_section(plan_data.get("agents"), "agents")
        tasks_data = _extract_section(plan_data.get("tasks"), "tasks")
        crew_data = plan_data.get("crew", {})
    else:
        agents_data = _extract_section(_yaml_load(plan_dir / "agents.yaml"), "agents")
        tasks_data = _extract_section(_yaml_load(plan_dir / "tasks.yaml"), "tasks")
        crew_data = _yaml_load(plan_dir / "crew.yaml") or {}

    payload = {
        "summary": crew_data.get("summary", ""),
        "process": crew_data.get("process", "sequential"),
        "agents": agents_data,
        "tasks": tasks_data,
    }

    plan = CrewPlan.model_validate(payload)

    prompt: Optional[str] = None
    metadata_path = plan_dir / "metadata.yaml"
    if metadata_path.exists():
        try:
            metadata = _yaml_load(metadata_path) or {}
            if isinstance(metadata, dict):
                raw_prompt = metadata.get("prompt")
                if isinstance(raw_prompt, str) and raw_prompt.strip():
                    prompt = raw_prompt
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to parse metadata for %s: %s", plan_dir, exc)

    return plan, prompt


@dataclass
class SavedPlanRecord:
    path: Path
    plan: CrewPlan
    prompt: Optional[str]
    created_at: datetime
    storage_format: str = "json"


def _format_plan_label(record: SavedPlanRecord) -> str:
    timestamp = record.created_at.strftime("%Y-%m-%d %H:%M:%S")
    summary = record.plan.summary or "(no summary)"
    return f"{timestamp} · {record.path.name} · {summary}"


def _discover_saved_plans(outdir: Path) -> list[SavedPlanRecord]:
    plans: list[SavedPlanRecord] = []

    for plan_dir in sorted(
        (p for p in outdir.glob("plan_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            plan, prompt = _read_plan_bundle(plan_dir)
        except Exception as exc:  # pragma: no cover - defensive IO handling
            logging.warning("Could not read saved plan bundle %s: %s", plan_dir, exc)
            continue

        created_at = datetime.fromtimestamp(plan_dir.stat().st_mtime)
        plans.append(
            SavedPlanRecord(
                path=plan_dir,
                plan=plan,
                prompt=prompt,
                created_at=created_at,
                storage_format="yaml",
            )
        )

    for plan_path in sorted(
        outdir.glob("plan_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            with plan_path.open("r", encoding="utf-8") as fh:
                raw_data = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive IO handling
            logging.warning("Could not read saved plan %s: %s", plan_path, exc)
            continue

        plan_data: Optional[dict[str, Any]] = None
        prompt: Optional[str] = None

        if isinstance(raw_data, dict):
            if "plan" in raw_data and isinstance(raw_data["plan"], dict):
                plan_data = raw_data["plan"]
                prompt = raw_data.get("prompt") or raw_data.get("_prompt")
            else:
                prompt = raw_data.get("_prompt")
                if "_prompt" in raw_data:
                    plan_data = {k: v for k, v in raw_data.items() if k != "_prompt"}
                else:
                    plan_data = raw_data

        if not plan_data:
            logging.warning("Ignoring malformed plan file: %s", plan_path)
            continue

        try:
            plan = CrewPlan.model_validate(plan_data)
        except Exception as exc:  # pragma: no cover - validation is already tested elsewhere
            logging.warning("Failed to parse plan %s: %s", plan_path, exc)
            continue

        created_at = datetime.fromtimestamp(plan_path.stat().st_mtime)
        plans.append(
            SavedPlanRecord(
                path=plan_path,
                plan=plan,
                prompt=prompt,
                created_at=created_at,
                storage_format="json",
            )
        )

    return plans


def _select_plan_with_curses(
    plans: list[SavedPlanRecord],
) -> tuple[Optional[SavedPlanRecord], bool]:
    try:
        import curses
    except Exception:  # pragma: no cover - curses missing on some platforms
        return None, False

    if not plans:
        return None, False

    selected: dict[str, int] = {"index": 0}

    def _menu(stdscr):
        curses.curs_set(0)
        idx = selected["index"]
        offset = 0

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            instructions = "Use ↑/↓ to select a plan. Enter executes, q cancels."
            stdscr.addnstr(0, 0, instructions, max(0, width - 1))

            visible_rows = max(1, height - 2)
            if idx < offset:
                offset = idx
            elif idx >= offset + visible_rows:
                offset = idx - visible_rows + 1

            for row in range(visible_rows):
                plan_idx = offset + row
                if plan_idx >= len(plans):
                    break
                record = plans[plan_idx]
                label = _format_plan_label(record)
                prefix = "➤ " if plan_idx == idx else "  "
                text = (prefix + label)[: max(0, width - 1)]
                if plan_idx == idx:
                    stdscr.attron(curses.A_REVERSE)
                    stdscr.addnstr(row + 1, 0, text, max(0, width - 1))
                    stdscr.attroff(curses.A_REVERSE)
                else:
                    stdscr.addnstr(row + 1, 0, text, max(0, width - 1))

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(plans)
            elif key in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(plans)
            elif key in (curses.KEY_ENTER, 10, 13):
                selected["index"] = idx
                return
            elif key in (27, ord("q")):
                raise KeyboardInterrupt

    try:
        import curses

        curses.wrapper(_menu)
    except KeyboardInterrupt:
        return None, True
    except curses.error:  # pragma: no cover - terminal limitations
        return None, False

    return plans[selected.get("index", 0)], False


def _select_plan_via_input(
    plans: list[SavedPlanRecord],
) -> tuple[Optional[SavedPlanRecord], bool]:
    if not plans:
        return None, False

    print("\nSaved plans available:")
    for idx, record in enumerate(plans, start=1):
        label = _format_plan_label(record)
        print(f" {idx}. {label}")

    while True:
        try:
            response = input("Select a plan number to execute (empty to cancel): ")
        except EOFError:
            return None, True

        if not response.strip():
            return None, True

        if response.strip().lower() in {"q", "quit", "exit"}:
            return None, True

        try:
            index = int(response.strip()) - 1
        except ValueError:
            print("Invalid selection. Please enter a number from the list.")
            continue

        if 0 <= index < len(plans):
            return plans[index], False

        print("Selection out of range. Try again.")


def _select_saved_plan(outdir: Path) -> tuple[Optional[SavedPlanRecord], bool]:
    plans = _discover_saved_plans(outdir)
    if not plans:
        logging.error("No saved plans were found in %s", outdir)
        return None, False

    if sys.stdin.isatty() and sys.stdout.isatty():
        selection, cancelled = _select_plan_with_curses(plans)
        if selection:
            return selection, False
        if cancelled:
            logging.info("Plan selection cancelled by user.")
            return None, True

    if not sys.stdin.isatty():
        logging.error("Interactive selection requires a TTY. Cannot continue.")
        return None, False

    selection, cancelled = _select_plan_via_input(plans)
    if selection:
        return selection, False
    if cancelled:
        logging.info("Plan selection cancelled by user.")
        return None, True

    return None, False


def _ensure_prompt_for_plan(record: SavedPlanRecord) -> Optional[str]:
    if record.prompt:
        return record.prompt

    if record.storage_format == "yaml" and record.path.is_dir():
        metadata_path = record.path / "metadata.yaml"
        if metadata_path.exists():
            try:
                metadata = _yaml_load(metadata_path) or {}
                if isinstance(metadata, dict):
                    stored_prompt = metadata.get("prompt")
                    if isinstance(stored_prompt, str) and stored_prompt.strip():
                        record.prompt = stored_prompt
                        return stored_prompt
            except Exception as exc:  # pragma: no cover - defensive IO handling
                logging.warning("Failed to read prompt metadata for %s: %s", record.path, exc)

    if not sys.stdin.isatty():
        logging.error(
            "The selected plan (%s) does not include the original prompt and no TTY is available to ask for it.",
            record.path,
        )
        return None

    try:
        response = input(
            f"Enter the original prompt for {record.path.name} (required to execute the crew): "
        )
    except EOFError:
        logging.error("Prompt capture interrupted. Aborting execution.")
        return None

    prompt = response.strip()
    if not prompt:
        logging.error("A prompt is required to run the selected crew.")
        return None

    _persist_prompt_for_plan(record, prompt)
    record.prompt = prompt

    return prompt


def _persist_prompt_for_plan(record: SavedPlanRecord, prompt: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")

    if record.storage_format == "yaml" and record.path.is_dir():
        metadata_path = record.path / "metadata.yaml"
        metadata: dict[str, Any]
        try:
            existing = _yaml_load(metadata_path) if metadata_path.exists() else {}
            metadata = existing if isinstance(existing, dict) else {}
        except Exception:
            metadata = {}

        metadata.update({"prompt": prompt, "updated_at": timestamp})
        _yaml_dump(metadata_path, metadata)
        return

    if record.path.is_file() and record.path.suffix == ".json":
        try:
            with record.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                raw["_prompt"] = prompt
                with record.path.open("w", encoding="utf-8") as fh:
                    json.dump(raw, fh, indent=2, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive IO handling
            logging.warning("Failed to persist prompt for %s: %s", record.path, exc)


def _save_outputs(outdir: Path, plan: CrewPlan, result, fmt: str, prompt: Optional[str]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_dir = outdir / f"plan_{timestamp}"
    _write_plan_bundle(plan_dir, plan, prompt)

    if fmt == "json":
        result_file = plan_dir / "result.json"
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
        result_file = plan_dir / "result.txt"
        with result_file.open("w", encoding="utf-8") as fh:
            if isinstance(result, (dict, list)):
                fh.write(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                fh.write(str(result))

    logging.info("Outputs saved in %s (result: %s)", plan_dir, result_file)


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
        outdir = _prepare_output_dir(args.output_dir)

        run_existing_plan = args.execute and not args.prompt and not args.prompt_file

        prompt: Optional[str] = None
        plan: Optional[CrewPlan] = None
        built: Optional[BuiltCrew] = None

        if run_existing_plan:
            if args.dry_run:
                logging.error("--dry-run cannot be combined with executing a saved plan.")
                return 2

            selection, cancelled = _select_saved_plan(outdir)
            if not selection:
                return 0 if cancelled else 1

            plan = selection.plan
            prompt = _ensure_prompt_for_plan(selection)
            if not prompt:
                return 1

            built = orchestrator.build_crew(plan)
            should_run = True
        else:
            prompt = _load_prompt(args)

            if args.dry_run:
                plan = orchestrator.plan(prompt)
                if args.show_plan:
                    print(_plan_to_yaml_text(plan))
                _save_outputs(outdir, plan, {}, "json", prompt)
                return 0

            built = orchestrator.plan_and_build(prompt)
            plan = built.plan

            if args.execute:
                should_run = True
            else:
                should_run = _prompt_for_execution(plan)

        assert plan is not None and built is not None and prompt is not None

        if args.show_plan:
            print("\n=== Plan (YAML) ===\n")
            print(_plan_to_yaml_text(plan))

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

        _save_outputs(outdir, plan, saved_result, args.format, prompt)
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
