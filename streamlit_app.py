"""
LIDARis - Streamlit Cloud Default Entrypoint.
Delegates to app.app.main() ensuring repository root is in sys.path.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.app import main

if __name__ == "__main__":
    main()
