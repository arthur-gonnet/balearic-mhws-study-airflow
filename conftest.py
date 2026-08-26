import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for path in (ROOT / "src", ROOT / "dags", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
