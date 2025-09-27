'''
-----------------------------------
OS agnostic path finder
-----------------------------------
'''

from __future__ import annotations
import os
from pathlib import Path
from typing import Union

# --- Resolve $WORK with sane fallbacks ---------------------------------------
def get_work_dir() -> Path:
    """Return the active WORK directory.
    Priority:
      1) $WORK if set
      2) .env value if python-dotenv loaded it into env
      3) $HOME/palma_work as a safe local default
    """
    work = os.environ.get("WORK")
    if not work or not work.strip():
        home = os.environ.get("HOME", str(Path("~").expanduser()))
        work = str(Path(home) / "palma_work")
    return Path(work).expanduser().resolve()

WORK = get_work_dir()
REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Helpers ------------------------------------------------------------------
def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def work_path(*parts: str) -> Path:
    """Join under WORK."""
    return WORK.joinpath(*parts)

def repo_path(*parts: str) -> Path:
    """Join under the repo root (for code, configs)."""
    return REPO_ROOT.joinpath(*parts)

# Conventional locations under WORK
def training_data_path(example: str) -> Path:
    return ensure_dir(work_path("training_data", example))

def state_dicts_path(example: str) -> Path:
    return ensure_dir(work_path("state_dicts", example))

def benchmarks_path(example: str) -> Path:
    return ensure_dir(work_path("benchmarks", example))
