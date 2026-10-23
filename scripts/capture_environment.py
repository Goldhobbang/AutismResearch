"""Record reproducibility metadata without inspecting secrets."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
packages = {}
for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'torch']:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
try:
    git = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root,
                                  text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    git = None
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
              executable=sys.executable, platform=platform.platform(),
              processor=platform.processor(), packages=packages, seed=42, git_commit=git)
target = root/'results/logs/environment_current.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))
