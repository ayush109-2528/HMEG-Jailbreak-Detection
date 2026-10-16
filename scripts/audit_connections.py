"""Check local Python/TypeScript import targets; never scan secrets or environments."""
import ast,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
errors=[];files=[]
for folder in ['app','database','ml','scripts']:
 for path in (ROOT/'backend'/folder).rglob('*.py'):
  files.append(str(path.relative_to(ROOT)))
  package=list(path.relative_to(ROOT/'backend').parts[:-1])
  for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
   targets=[]
   if isinstance(node,ast.Import): targets=[n.name for n in node.names]
   elif isinstance(node,ast.ImportFrom):
    prefix=package[:len(package)-node.level+1] if node.level else []
    targets=['.'.join(prefix+((node.module or '').split('.') if node.module else []))]
   for target in targets:
    if target.split('.')[0] not in ['app','database','ml','scripts']:continue
    resolved=ROOT/'backend'/Path(*target.split('.'))
    if not resolved.with_suffix('.py').exists() and not (resolved/'__init__.py').exists(): errors.append(f'{path.name}: unresolved {target}')
for path in (ROOT/'frontend/src').rglob('*'):
 if path.suffix not in ['.ts','.tsx']:continue
 files.append(str(path.relative_to(ROOT)))
 for target in re.findall(r"(?:from\s+|import\s*\(?)[\"'](\.[^\"']+)[\"']",path.read_text(encoding='utf-8')):
  p=path.parent/target
  if not any(x.exists() for x in [p,Path(str(p)+'.ts'),Path(str(p)+'.tsx'),p/'index.ts',p/'index.tsx']):errors.append(f'{path.name}: unresolved {target}')
print(json.dumps({'source_files_checked':len(files),'errors':errors,'files':files},indent=2))
raise SystemExit(bool(errors))
