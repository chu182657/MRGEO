"""Restore exact archived data locally; no downloads, API calls or credential extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=['deepseek','claude','gemini','gpt4o_mini'])
    p.add_argument('--archive', type=Path, required=True)
    args = p.parse_args()
    expected = json.loads((ROOT/'audit'/f'{args.variant}_data_inventory.json').read_text())['file_sha256']
    with zipfile.ZipFile(args.archive) as z:
        names = [n for n in z.namelist() if n.replace('\\','/').endswith('/data/samples.json') and '/.venv/' not in n]
        if len(names) != 1:
            raise SystemExit('Expected exactly one data/samples.json in archive.')
        blob = z.read(names[0])
    if hashlib.sha256(blob).hexdigest() != expected:
        raise SystemExit('Data hash differs from the audited upload. No file written.')
    target = ROOT/'experiments'/args.variant/'data'/'samples.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != blob:
        raise SystemExit('Different local data already exists. No file overwritten.')
    target.write_bytes(blob)
    print(f'Restored {target}. Historical data: all query fields are empty; not suitable for a new claimed query-conditioned experiment.')
if __name__ == '__main__':
    main()
