"""Offline release checks: syntax, credential patterns and archive size. Makes no API calls."""
import ast
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[1]
errors=[]
files=list(ROOT.rglob('*.py'))
for p in files:
    try: ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    except SyntaxError as e: errors.append(f'Syntax: {p.relative_to(ROOT)}:{e.lineno}')
for p in ROOT.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts: continue
    if p.stat().st_size > 20*1024*1024: errors.append(f'Large file: {p.relative_to(ROOT)}')
    if p.suffix in {'.py','.md','.json','.csv','.txt','.example'}:
        text=p.read_text(encoding='utf-8-sig')
        if re.search(r'sk-' + r'[A-Za-z0-9_-]{16,}', text): errors.append(f'Credential pattern: {p.relative_to(ROOT)}')
if errors:
    print('\n'.join(errors));raise SystemExit(1)
print(f'PASS: {len(files)} Python files parse; no scanned token pattern; no file over 20 MiB.')
print('This check does not validate scientific reproducibility or API availability.')
