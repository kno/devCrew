"""Interactive helpers to choose and prepare saved plans."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from .plan_storage import (
    SavedPlanRecord,
    discover_saved_plans,
    format_plan_label,
    persist_prompt_for_plan,
    resolve_saved_prompt,
)


LOGGER = logging.getLogger(__name__)


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
                label = format_plan_label(record)
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
        label = format_plan_label(record)
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


def select_saved_plan(outdir: Path) -> tuple[Optional[SavedPlanRecord], bool]:
    plans = discover_saved_plans(outdir)
    if not plans:
        LOGGER.error("No saved plans were found in %s", outdir)
        return None, False

    if sys.stdin.isatty() and sys.stdout.isatty():
        selection, cancelled = _select_plan_with_curses(plans)
        if selection:
            return selection, False
        if cancelled:
            LOGGER.info("Plan selection cancelled by user.")
            return None, True

    if not sys.stdin.isatty():
        LOGGER.error("Interactive selection requires a TTY. Cannot continue.")
        return None, False

    selection, cancelled = _select_plan_via_input(plans)
    if selection:
        return selection, False
    if cancelled:
        LOGGER.info("Plan selection cancelled by user.")
        return None, True

    return None, False


def ensure_prompt_for_plan(record: SavedPlanRecord) -> Optional[str]:
    prompt = resolve_saved_prompt(record)
    if prompt:
        record.prompt = prompt
        return prompt

    if not sys.stdin.isatty():
        LOGGER.error(
            "The selected plan (%s) does not include the original prompt and no TTY is available to ask for it.",
            record.path,
        )
        return None

    try:
        response = input(
            f"Enter the original prompt for {record.path.name} (required to execute the crew): "
        )
    except EOFError:
        LOGGER.error("Prompt capture interrupted. Aborting execution.")
        return None

    prompt = response.strip()
    if not prompt:
        LOGGER.error("A prompt is required to run the selected crew.")
        return None

    persist_prompt_for_plan(record, prompt)
    record.prompt = prompt
    return prompt


__all__ = [
    "ensure_prompt_for_plan",
    "select_saved_plan",
]
