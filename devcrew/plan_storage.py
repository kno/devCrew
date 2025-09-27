"""Utilities for persisting crew plans and execution artefacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import yaml

from .orchestrator import CrewPlan


LOGGER = logging.getLogger(__name__)


def prepare_output_dir(path: str | Path) -> Path:
    """Ensure the output directory exists and return it as a :class:`Path`."""

    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _yaml_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def _yaml_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def plan_to_payload(plan: CrewPlan) -> dict[str, Any]:
    """Serialise a :class:`CrewPlan` into a JSON/YAML friendly dictionary."""

    agents = [agent.model_dump() for agent in plan.agents]
    tasks = [task.model_dump() for task in plan.tasks]
    return {
        "crew": {
            "summary": plan.summary,
            "process": plan.process,
            "agents": [agent.get("name") for agent in agents],
            "tasks": [task.get("name") for task in tasks],
        },
        "agents": agents,
        "tasks": tasks,
    }


def plan_to_yaml_text(plan: CrewPlan) -> str:
    return yaml.safe_dump(plan_to_payload(plan), sort_keys=False, allow_unicode=True)


def update_plan_metadata(plan_dir: Path, **updates: Any) -> None:
    """Merge ``updates`` into the plan metadata file."""

    metadata_path = plan_dir / "metadata.yaml"
    try:
        existing = _yaml_load(metadata_path) if metadata_path.exists() else {}
        metadata = existing if isinstance(existing, dict) else {}
    except Exception:  # pragma: no cover - defensive IO handling
        metadata = {}

    clean_updates = {key: value for key, value in updates.items() if value is not None}
    if not clean_updates and metadata_path.exists():
        return

    metadata.update(clean_updates)
    _yaml_dump(metadata_path, metadata)


def write_plan_bundle(plan_dir: Path, plan: CrewPlan, prompt: Optional[str]) -> None:
    bundle = plan_to_payload(plan)
    plan_dir.mkdir(parents=True, exist_ok=True)

    _yaml_dump(plan_dir / "plan.yaml", bundle)
    _yaml_dump(plan_dir / "agents.yaml", {"agents": bundle["agents"]})
    _yaml_dump(plan_dir / "tasks.yaml", {"tasks": bundle["tasks"]})
    _yaml_dump(plan_dir / "crew.yaml", bundle["crew"])

    update_plan_metadata(
        plan_dir,
        saved_at=datetime.now().isoformat(timespec="seconds"),
        prompt=prompt,
    )


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


def read_plan_bundle(plan_dir: Path) -> tuple[CrewPlan, Optional[str]]:
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
            LOGGER.warning("Failed to parse metadata for %s: %s", plan_dir, exc)

    return plan, prompt


@dataclass
class SavedPlanRecord:
    path: Path
    plan: CrewPlan
    prompt: Optional[str]
    created_at: datetime
    storage_format: str = "json"


def format_plan_label(record: SavedPlanRecord) -> str:
    timestamp = record.created_at.strftime("%Y-%m-%d %H:%M:%S")
    summary = record.plan.summary or "(no summary)"
    return f"{timestamp} · {record.path.name} · {summary}"


def discover_saved_plans(outdir: Path) -> list[SavedPlanRecord]:
    plans: list[SavedPlanRecord] = []

    for plan_dir in sorted(
        (candidate for candidate in outdir.glob("plan_*") if candidate.is_dir()),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    ):
        try:
            plan, prompt = read_plan_bundle(plan_dir)
        except Exception as exc:  # pragma: no cover - defensive IO handling
            LOGGER.warning("Could not read saved plan bundle %s: %s", plan_dir, exc)
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
        outdir.glob("plan_*.json"), key=lambda candidate: candidate.stat().st_mtime, reverse=True
    ):
        try:
            with plan_path.open("r", encoding="utf-8") as fh:
                raw_data = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive IO handling
            LOGGER.warning("Could not read saved plan %s: %s", plan_path, exc)
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
                    plan_data = {key: value for key, value in raw_data.items() if key != "_prompt"}
                else:
                    plan_data = raw_data

        if not plan_data:
            LOGGER.warning("Ignoring malformed plan file: %s", plan_path)
            continue

        try:
            plan = CrewPlan.model_validate(plan_data)
        except Exception as exc:  # pragma: no cover - validation tested elsewhere
            LOGGER.warning("Failed to parse plan %s: %s", plan_path, exc)
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


def initialise_plan_storage(
    outdir: Path,
    plan: CrewPlan,
    prompt: Optional[str],
    *,
    source: Optional[Path] = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:6]
    plan_dir = outdir / f"plan_{timestamp}_{suffix}"

    write_plan_bundle(plan_dir, plan, prompt)

    now_iso = datetime.now().isoformat(timespec="seconds")
    update_plan_metadata(
        plan_dir,
        status="planned",
        executed=False,
        plan_id=plan_dir.name,
        source=str(source) if source else None,
        initialised_at=now_iso,
        updated_at=now_iso,
    )
    LOGGER.info("Plan saved in %s", plan_dir)
    return plan_dir


def persist_plan_result(
    plan_dir: Path,
    plan: CrewPlan,
    result: Any,
    fmt: str,
    status: str,
) -> Path:
    plan_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        result_file = plan_dir / "result.json"
        payload = {
            "summary": plan.summary,
            "process": plan.process,
            "agents": [agent.name for agent in plan.agents],
            "tasks": [task.name for task in plan.tasks],
            "result": result,
        }
        with result_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    else:
        result_file = plan_dir / "result.txt"
        with result_file.open("w", encoding="utf-8") as fh:
            if isinstance(result, (dict, list)):
                fh.write(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                fh.write(str(result))

    update_plan_metadata(
        plan_dir,
        status=status,
        executed=status == "executed",
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )
    plan_file = plan_dir / "plan.yaml"
    LOGGER.info("Outputs saved: %s , %s", plan_file, result_file)
    return result_file


def _load_metadata(plan_dir: Path) -> Optional[dict[str, Any]]:
    metadata_path = plan_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None
    try:
        metadata = _yaml_load(metadata_path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to read metadata for %s: %s", plan_dir, exc)
        return None
    return metadata if isinstance(metadata, dict) else None


def resolve_saved_prompt(record: SavedPlanRecord) -> Optional[str]:
    if record.prompt:
        return record.prompt
    if record.storage_format == "yaml" and record.path.is_dir():
        metadata = _load_metadata(record.path)
        if metadata:
            prompt = metadata.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                return prompt
    return None


def persist_prompt_for_plan(record: SavedPlanRecord, prompt: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")

    if record.storage_format == "yaml" and record.path.is_dir():
        update_plan_metadata(
            record.path,
            prompt=prompt,
            updated_at=timestamp,
        )
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
            LOGGER.warning("Failed to persist prompt for %s: %s", record.path, exc)


def dump_plan_on_failure(plan: CrewPlan, plan_dir: Optional[Path]) -> Path:
    """Persist ``plan`` when an unexpected error happens during execution."""

    if plan_dir is None:
        target_dir = Path.cwd()
    else:
        target_dir = plan_dir
        target_dir.mkdir(parents=True, exist_ok=True)

    dump_path = target_dir / "plan_on_error.json"
    with dump_path.open("w", encoding="utf-8") as fh:
        json.dump(plan_to_payload(plan), fh, indent=2, ensure_ascii=False)
    LOGGER.error("Stored plan snapshot after failure: %s", dump_path)
    return dump_path


__all__ = [
    "SavedPlanRecord",
    "discover_saved_plans",
    "dump_plan_on_failure",
    "format_plan_label",
    "initialise_plan_storage",
    "persist_plan_result",
    "persist_prompt_for_plan",
    "plan_to_payload",
    "plan_to_yaml_text",
    "prepare_output_dir",
    "resolve_saved_prompt",
    "update_plan_metadata",
    "write_plan_bundle",
]
