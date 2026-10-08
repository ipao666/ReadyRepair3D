#!/usr/bin/env python3
"""Run the documented lightweight test profile, then audit committed evidence."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "01_项目代码"
TESTS = [
    "test_ready3d_v3_data_protocol.py", "test_ready3d_v3_ranking.py",
    "test_evaluate_ready3d_v3_strategies.py", "test_quality_scoring.py",
    "test_aggregate_final_results.py", "test_submission_metrics_contract.py",
    "test_release_hashes.py", "test_release_code_contract.py", "sana_lora/test_weights.py",
    "sana_lora/test_data_protocol.py", "sana_lora/test_train_config.py",
    "sana_lora/test_evaluate_candidates.py", "sana_lora/test_prompt_splits.py",
]


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(CODE), str(CODE / "src")))
    commands = [
        ([sys.executable, "-m", "pytest", "-q", "--import-mode=importlib",
          *[f"tests/{test}" for test in TESTS]], CODE),
        ([sys.executable, "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py"], ROOT),
        ([sys.executable, "tools/verify_portfolio.py"], ROOT),
    ]
    for command, directory in commands:
        completed = subprocess.run(command, cwd=directory, env=env)
        if completed.returncode:
            return completed.returncode
    print("CPU profile passed. GPU generation, training and full integration tests were not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
