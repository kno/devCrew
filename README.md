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

2. Create a `.env` file with the credentials for the language models you want to
   use. For example:

   ```bash
   echo "MODEL_PROVIDER=ollama" >> .env
   echo "PLANNER_MODEL=llama3" >> .env
   echo "AGENT_MODEL=llama3" >> .env
   ```

   Any variables defined in `.env` are loaded automatically (such as
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST`, etc.) so you can switch
   between providers without changing the code.

3. Run the orchestrator by passing the problem statement as an argument:

   ```bash
   python -m devcrew "Design a marketing strategy for a new eco-friendly water bottle"
   ```

   Once the crew is planned you'll be prompted to confirm whether it should be
   executed. Use `--execute` to skip the prompt and immediately run the crew,
   `--show-plan` to inspect the planned crew and tasks before execution in
   standard crewAI YAML form, or `--dry-run` to only produce the plan without
   running the crew. You can also override providers/models via CLI flags, e.g.

   ```bash
   python -m devcrew --planner-provider anthropic --planner-model claude-3-haiku \
     "Summarise the latest quarterly results for stakeholders"
   ```

   If you already have plans stored in the `outputs/` directory you can re-run
   any of them without providing a new prompt. Launch the CLI with
   `python -m devcrew --execute` and you'll be presented with an interactive
   list of saved plans. Use the arrow keys (or number selection fallback) to
   choose the plan you want to execute. The orchestrator persists each plan in a
   `plan_<timestamp>/` folder containing crewAI-compatible YAML files:

   - `plan.yaml` – consolidated snapshot including summary, process, agents and
     tasks.
   - `agents.yaml` and `tasks.yaml` – separate definitions for each agent and
     task so they can be inspected or reused independently.
   - `crew.yaml` – high-level metadata (summary and process) referencing the
     planned agents and tasks.
   - `metadata.yaml` – auxiliary information such as the original prompt.

   When a saved plan is executed the CLI rebuilds the crew directly from these
   YAML files, ensuring that the stored configuration is the one actually run.
   Plans are now written to disk immediately after the planner responds so the
   crew definition is preserved even if execution is skipped or fails later.
   As soon as a run finishes (or is skipped) the corresponding directory is
   updated with the execution status (`planned`, `executed`, `skipped`,
   `dry-run`, `cancelled`, or `error`) and a `result.json`/`result.txt` file so
   every piece of data is captured the moment it becomes available. If a run
   fails unexpectedly the CLI now dumps the YAML plan to the error log (and
   points to the saved bundle) to simplify post-mortem analysis.

## Extending

To add new tools register them in the `ToolRegistry` within `devcrew/tools.py`
and include their names in the planner instructions so the planning agent can
assign them to relevant agents or tasks.
