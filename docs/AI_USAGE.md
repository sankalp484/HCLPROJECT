# AI-usage disclosure

AI coding assistants were used, as the guide allows. Edit this file so it matches what
each member actually did.

| Part | Assistant | How it was verified |
|---|---|---|
| Loader, validation, cost/carbon models, solver, analysis tools, LangGraph graph, API, UI, eval harness, tests, Docker files, README | Claude Code (Anthropic), working from the participant guide and our data | 59 pytest tests (including a 4-order dataset solved by hand and API contract tests); 32-case evaluation with expected results from the data card and hand-solved data; an independent plan checker that re-derives capacity, eligibility and delivery from the raw tables; manual review of solver output on the data pack |
| Synthetic data pack | (team: state the tool and prompts) | `scripts/validate_data.py`: all checks pass, including row counts against the data card |

Every member must be able to explain the code they submit. The README's "Why this shape"
and "The model" sections are the shared explanation.
