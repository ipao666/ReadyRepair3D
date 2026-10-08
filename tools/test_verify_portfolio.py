import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_portfolio import ROOT, validate_strategies, validate_final, verify, verify_asset


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "frozen_results/stage01_ready3d_v3_r2/strategy_metrics.json").read_text(encoding="utf-8"))

    def test_committed_evidence_and_assets(self):
        self.assertEqual(verify(ROOT)["showcase_assets_verified"], 3)

    def test_rejects_duplicate_group(self):
        self.data["group_results"].append(copy.deepcopy(self.data["group_results"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate or missing"):
            validate_strategies(self.data)

    def test_rejects_modified_mean(self):
        self.data["strategies"]["ready_v3_top1"]["mean_quality"] += 0.1
        with self.assertRaisesRegex(ValueError, "mean_quality"):
            validate_strategies(self.data)

    def test_rejects_nonfinite_score(self):
        self.data["group_results"][0]["selected_quality"] = float("nan")
        with self.assertRaisesRegex(ValueError, "quality range"):
            validate_strategies(self.data)

    def test_rejects_inconsistent_final_difference(self):
        final = json.loads((ROOT / "frozen_results/stage06_final_test/final_comparison.json").read_text(encoding="utf-8"))
        final["lora_summary"]["mean_q"] += 0.1
        with self.assertRaisesRegex(ValueError, "final mean difference"):
            validate_final(final)

    def test_rejects_corrupted_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset.glb").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_asset(root, "asset.glb", "0" * 64)


if __name__ == "__main__":
    unittest.main()
