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
   `--show-plan` to inspect the planned crew and tasks before execution or
   `--dry-run` to only produce the plan without running the crew. You can also
   override providers/models via CLI flags, e.g.

   ```bash
   python -m devcrew --planner-provider anthropic --planner-model claude-3-haiku \
     "Summarise the latest quarterly results for stakeholders"
   ```

## Extending

To add new tools register them in the `ToolRegistry` within `devcrew/tools.py`
and include their names in the planner instructions so the planning agent can
assign them to relevant agents or tasks.
