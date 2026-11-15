import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import xgb_training  # noqa: E402

if __name__ == "__main__":
    xgb_training.main("UO")
