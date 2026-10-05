import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("PH_DATA_ROOT", os.path.join(REPO, "data"))  # tests that need the data skip without it
sys.path.insert(0, REPO)
