"""
V7 and V8 Preflight Health Test Suite
Runs clean verification tests for the active production pipelines.
"""

import sys
from pathlib import Path

# Add project root and v7 to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "v7"))

from preflight_health_check import run_preflight_check

if __name__ == "__main__":
    success = run_preflight_check()
    if success:
        print("\n>>> ALL V7 & V8 HEALTH CHECKS PASSED SUCCESSFULLY! <<<")
        sys.exit(0)
    else:
        print("\n>>> PREFLIGHT HEALTH CHECKS FAILED! <<<")
        sys.exit(1)
