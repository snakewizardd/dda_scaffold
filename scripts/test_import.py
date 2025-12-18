"""Test that all modules can be imported"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing DDA-X imports...")
print("-" * 40)

try:
    # Core imports
    from src.core import DDAState, ActionDirection, ForceAggregator
    print("[OK] Core modules")

    # Search imports
    from src.search import DDASearchTree, DDAMCTS, ValueEstimator
    print("[OK] Search modules")

    # Memory imports
    from src.memory import ExperienceLedger, LedgerEntry
    print("[OK] Memory modules")

    # Agent import
    from src.agent import DDAXAgent, DDAXConfig
    print("[OK] Agent module")

    # Package-level imports
    import src
    print(f"[OK] Package import (version {src.__version__})")

    print("-" * 40)
    print("All imports successful!")

except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)