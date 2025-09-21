# devCrew

A dynamic crewAI orchestrator capable of planning and executing on-demand crews
based on a single natural language prompt.

## Features

- Uses a dedicated planning LLM to analyse the incoming problem and propose the
  required agents, tasks and tool assignments.
- Builds the crew programmatically from the produced plan and executes it using
  `crewai`'s `Crew` runtime.
- Ships with a minimal `calculator` tool and a registry that can be expanded
  with custom tools.
- Offers a command line interface that can preview the generated plan (`--dry-run`)
  or run the crew end-to-end.

## Getting started

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your LLM credentials (for example `OPENAI_API_KEY`) in the
   environment so that `langchain`/`crewai` can access them.

3. Run the orchestrator by passing the problem statement as an argument:

   ```bash
   python -m devcrew "Design a marketing strategy for a new eco-friendly water bottle"
   ```

   Use `--show-plan` to inspect the planned crew and tasks before execution or
   `--dry-run` to only produce the plan without running the crew.

## Extending

To add new tools register them in the `ToolRegistry` within `devcrew/tools.py`
and include their names in the planner instructions so the planning agent can
assign them to relevant agents or tasks.
