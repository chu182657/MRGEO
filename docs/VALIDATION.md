# Validation performed

- All 106 Python files parsed successfully under Python 3.12. One existing docstring emits a nonfatal invalid-escape warning.
- Four variants: 14 prompt builders per variant executed on a synthetic dictionary-form document; all returned strings. No LLM was called. Claude expects dictionary-form documents; string documents are not supported by that variant.
- All four configuration modules imported without real credentials. This checks offline accessibility, not live API authentication.
- Exact local data restoration tested successfully using the Gemini archive, including SHA-256 verification. A later local DeepSeek ZIP copy was truncated and rejected as an invalid ZIP; its original extracted data had already been inventoried. No truncated archive is distributed.
- DeepSeek `run_report_full23.py` ran to completion on the archived 2,800-row full23_200 numeric snapshot: three CSV tables and four PNGs were generated in a temporary output directory. These use legacy aggregation including failure rows. They are not validated manuscript results and are not distributed as final figures. Arial was unavailable and Matplotlib used a fallback font.
- Release source scanned for common API-key, GitHub-token, AWS-key and private-key patterns; no matches found. Original environment files, caches, logs and virtual environments excluded.
- Raw score rows were retained numerically; no failed-row removal, duplicate resolution, reweighting, significance testing or claimed ranking correction was performed.
- No live API calls, model availability tests, paid experiments, full dependency installation or end-to-end scientific reproduction performed. Network packages are not installed in this test environment, so API entry-point imports were not all exercised.
- The three supplied standalone plotting scripts were syntax-checked; their embedded statistics were not validated against query-level results.
