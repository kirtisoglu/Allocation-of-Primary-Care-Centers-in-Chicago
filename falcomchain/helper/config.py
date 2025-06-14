
from pathlib import Path

def find_project_root(marker=".git", max_depth=10):
    path = Path(__file__).resolve().parent
    for _ in range(max_depth):
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Could not find project root marker '{marker}'")

# cache it for reuse
PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "prepared_data"


#To get the path, use 
"""from helper.config import DATA_DIR
data_path = DATA_DIR / "initial.pkl"""

# or put the following in a yourlib.config.json or config.yaml file
"""from pathlib import Path

def find_project_root(marker="yourlib.config.json") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"No {marker} found in parent directories.")"""


# For a user configurable option, put the following in yourlib/config.py
"""import os
PROJECT_ROOT = os.environ.get("YOURLIB_ROOT", default=Path(__file__).parent)"""


# Adding CLI flag or API override is more flexible