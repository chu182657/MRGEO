# MRGEO — research code archive

This repository organizes the supplied experimental scripts, prompts, numeric score snapshots and plotting scripts. **It is an audited code archive, not a verified final reproduction of the manuscript.** Read [known limitations](docs/KNOWN_LIMITATIONS.md) before using any result. No new API experiments were run during packaging.

## Contents

| Path | Contents |
|---|---|
| `experiments/deepseek/` | Main generation, two evaluation versions, sequential/leave-one-out/order ablations, reports |
| `experiments/claude/` | Historical Claude variant with a different evaluation schema |
| `experiments/gemini/` | Gemini evaluation variant and reporting scripts |
| `experiments/gpt4o_mini/` | GPT-4o-mini variant, generation repair and full-metric evaluation |
| `experiments/*/preprocessing/` | Historical data preparation utilities; inspect inputs before running |
| `results/archived_scores/` | Numeric score snapshots, identifiers and error flags; not a curated final result set |
| `figures/supplied_statistics/` | Three user-supplied plotting scripts with embedded statistics |
| `audit/` | Original file hashes, inclusion/exclusion inventory, data and score audits |
| `tools/` | Added offline validation and exact local data restoration utilities |

The two Gemini uploads were byte-identical file by file; one copy is retained. Legacy method identifiers (`m2geo`, `MRGEOA-*`) remain unchanged to preserve joins with archived outputs. Folder names are provenance labels; they do not prove which engine generated or scored a given file.

## Install and validate offline

Use a fresh Python 3.10+ virtual environment. Dependencies are inferred from imports; the original dependency versions were not recovered as a validated environment lock.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/check_release.py
```

`check_release.py` parses sources without importing them or calling an API. Its passing result is not a scientific validation.

## Historical data

Full source texts, raw generations, local logs, environment files, virtual environments and binary output documents are not distributed here. The original uploads retain these files. No upstream data license was supplied; this package does not grant redistribution rights to third-party text. A public reproducible data release still requires a verified upstream source, license, preserved query/target fields and exact sample selection.

For the author to inspect the exact archived data locally:

```bash
python tools/prepare_local_data.py --variant deepseek --archive "/path/to/your-original-deepseek.zip"
```

Repeat with the matching archive and variant (`claude`, `gemini`, `gpt4o_mini`) as needed. The helper extracts only `data/samples.json`, verifies its hash and does not extract secrets. **All 1,000 supplied records have empty `query` fields. Do not treat this restoration as a repaired dataset.**

## API configuration and historical commands

Set credentials in shell environment variables; `.env.example` is documentation, not an automatically loaded configuration. Clients use the original variant-specific OpenAI-compatible endpoints. Provider access and historical model availability must be verified by the operator. Calls may incur costs.

```bash
# Linux/macOS
export API_KEY="YOUR_NEW_KEY"
# Windows PowerShell
# $env:API_KEY="YOUR_NEW_KEY"
```

Run each legacy script from its own variant directory because it uses local imports. Avoid combining variants into one directory. These commands describe existing entry points; fix the data/protocol limitations before launching a new scientific experiment.

```bash
cd experiments/deepseek
python run_generation.py --help
python run_generation_ablation_complete.py --help
python run_generation_leave_one_out.py --help
python run_generation_order_ablation.py --help
python run_evaluation_full23.py --help
```

Historical full-metric scoring explicitly needs `--max-samples 200` (DeepSeek defaults to 50):

```bash
python run_evaluation_full23.py --input /path/to/generation_results.jsonl --output /path/to/new_scores.csv --max-samples 200
```

For offline legacy reporting of an archived score snapshot:

```bash
python run_report_full23.py --input ../../results/archived_scores/deepseek/evaluation_results_full23_200.csv
```

This recreates the **legacy aggregation**, which can include failed rows; it is not a corrected manuscript table. Gemini/GPT use their own `run_evaluation_full23.py` and `run_report_full23.py`; Claude has `run_evaluation.py` and `run_report.py` and defaults to `ENABLE_JUDGE=false`. Do not use disabled-Judge scores as genuine model ratings.

## Figures

```bash
# From repository root:
python figures/supplied_statistics/plot_supplied_method_scores.py
python figures/supplied_statistics/plot_agreement_ci_v2.py
```

These render embedded values, not newly computed experimental statistics. Both CI figure versions retain their review-draft warning. Running v1/v2 writes the same figure basename; choose one version. These commands are for inspecting plots, not establishing the validity of the plotted values.

## Release status and license

Credentials were removed, runtime outputs separated from archived scores, and machine-specific default ablation output paths moved under each variant's output directory. Scientific prompts, evaluation formulas and legacy error handling were not rewritten. See [changes](docs/CHANGES.md) and [validation](docs/VALIDATION.md).

No software license has been selected on behalf of the authors. Before declaring an open-source release, the rights holders must add an appropriate `LICENSE` and resolve third-party notices. No author list, citation metadata, publication venue or repository URL has been invented. Add them once confirmed. Only claim public code availability after the repository is actually public and accessible.
