# Packaging changes

- Four model variants retained as separate runnable script directories; identical Gemini archive deduplicated.
- 101 experiment/preparation Python sources and three supplied plotting scripts preserved.
- API credential literals removed; configuration reads `API_KEY`, preparation utilities read `DEEPSEEK_API_KEY` (falling back to `API_KEY`).
- Empty-key import-time exceptions disabled for offline tooling. Existing clients retain their request-time credential handling. API instructions in historical comments may still refer to editing config; follow the top-level README instead.
- Optional `MRGEO_OUTPUT_DIR` added. Default ablation outputs moved from a personal Desktop directory to the variant output directory. Old command examples in historical docstrings are retained for provenance; use README commands.
- Claude disabled-Judge default preserved and made explicitly configurable with `ENABLE_JUDGE=true`.
- Prompts, method identifiers, scoring formulas, sampling rules and failure logic not changed.
- Numeric score CSV snapshots extracted separately from runtime outputs. Free-text queries/explanations/errors omitted when nonnumeric; error presence retained in `archived_judge_error`. Blank columns may remain. No scores recalculated or edited.
- Original hashes and disposition recorded. Original README files replaced by unified English documentation; generated binaries, virtual environments, logs, credentials, full source text and generations omitted.
- Added exact hash-checked local data restoration and offline syntax/credential checks. These are packaging tools, not original experimental code.
