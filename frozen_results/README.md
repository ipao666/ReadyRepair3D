# Frozen experiment records

These files are a compact, integrity-verifiable export of completed remote stages. They are not model artifacts or training data.

| Directory | Contents |
| --- | --- |
| `stage01_ready3d_v3_r2/` | Frozen Ready3D V3 Top-1 summary, aggregate and strategy metrics, calibration metadata, pseudo-label gate, and SHA-256 records |
| `stage02_sana_candidates/` | Frozen SANA candidate summary and the SHA-256 list for the complete generated candidate set |

`stage01_ready3d_v3_r2` supersedes the earlier Stage 1 freeze by separating the 3D scoring calibration from the Ready3D training calibration. All source assets remain on the secured remote environment and are intentionally excluded from Git.
