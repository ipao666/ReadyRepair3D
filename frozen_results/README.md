# Frozen experiment records

Compact exports of historical author experiments. These files are evidence, not a self-contained model release. Current interpretation and limitations: [中文 evidence guide](../docs/EVIDENCE.md).

| Directory | Included evidence |
|---|---|
| `stage01_ready3d_v3_r2/` | Six-strategy group records, aggregates, calibration metadata and historical gate |
| `stage02_sana_candidates/` | Generation summary and hash list; candidate images are not included |
| `stage05_lora_selection/` | Selected-checkpoint validation summary |
| `stage06_final_test/` | Final 64-group comparison summary and decision: `keep_base_sana` |

Stage 3/4 completion is described in the final report; full checkpoints and remote per-sample artifacts are not exported here. Stage 5/6 remote READY/hash artifacts are not all present in this repository. Hash lists may refer to remote files intentionally omitted from Git.

Stage 1 V3 did not beat the structural rule baseline. The final LoRA confidence interval crosses zero and adherence/technical validity declined. Do not turn these records into a claim of significant or universal improvement.

From the repository root, run `python tools/verify_portfolio.py` to recompute Stage 1 aggregates and validate the shipped showcase. Final Bootstrap cannot be independently recomputed without the missing group-level records.
