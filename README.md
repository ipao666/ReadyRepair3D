# ReadyRepair3D

Quality-aware text-to-image-to-3D generation and reconstruction research pipeline.

[中文说明](README.zh-CN.md) · [Frozen experiment records](frozen_results/README.md) · [冻结实验记录（中文）](frozen_results/README.zh-CN.md)

## Completed, frozen stages

| Stage | Status | Public repository contents |
| --- | --- | --- |
| Stage 0 — preparation | Frozen | Data protocol, split definition, manifests, and hashes |
| Stage 1 — Ready3D V3 Top-1 | Frozen | Evaluation metrics, calibration metadata, selection gate, and hashes |
| Stage 2 — SANA candidates | Frozen | Summary, candidate-coverage integrity hash list, and hashes |

Stage 3 (Hunyuan3D labels) and later LoRA training are intentionally not included: they are still running or not yet frozen.

## Reproducibility boundary

This repository contains source code, documentation, small manifests, frozen summaries, metrics, and SHA-256 integrity records. It deliberately excludes model weights, checkpoints, generated images, GLB meshes, caches, and remote runtime outputs. Obtain third-party models separately under their respective licenses; see [01_项目代码/THIRD_PARTY_MODELS.md](01_%E9%A1%B9%E7%9B%AE%E4%BB%A3%E7%A0%81/THIRD_PARTY_MODELS.md).

## Key Stage 1 result

On the held-out Stage 1 test partition, Ready3D V3 Top-1 selected candidates with mean quality **0.5353**, compared with **0.4546** for random Top-1, reducing mean regret from **0.1910** to **0.1103**. Model selection itself used validation only; the test partition was not used for selection.

## Repository layout

- `01_项目代码/` — implementation, tests, configuration, and technical documentation.
- `03_关键文档/` — project design and server handoff notes.
- `frozen_results/` — immutable records exported from completed remote stages.

## Status

The repository is private. The frozen records correspond to the completed Stage 0–2 work as of 2026-09-01.
