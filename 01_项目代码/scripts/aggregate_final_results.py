"""Aggregate frozen ReadyRepair3D experiment evidence for reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def summarize_repairs(report_paths: Iterable[Path]) -> dict[str, float | int]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    accepted = sum(bool(report.get("accepted")) for report in reports)
    rollback = sum(report.get("executed_action") == "rollback_original" for report in reports)
    passthrough = sum(report.get("executed_action") == "accept" for report in reports)
    modified = sum(
        report.get("executed_action") not in {"accept", "rollback_original"}
        for report in reports
    )
    modified_and_accepted = sum(
        bool(report.get("accepted"))
        and report.get("executed_action") not in {"accept", "rollback_original"}
        for report in reports
    )
    no_harm = 0
    loadable_after = 0
    gains: list[float] = []
    for report in reports:
        before = float(report.get("quality_before", {}).get("quality_score", 0.0))
        after_data = report.get("quality_after", {})
        after = float(after_data.get("quality_score", before))
        gains.append(after - before)
        no_harm += after + 1e-12 >= before
        loadable_after += all(
            bool(after_data.get(key))
            for key in ("trimesh_loadable", "blender_loadable", "finite")
        )
    count = len(reports)
    return {
        "report_count": count,
        "accepted_count": accepted,
        "accepted_rate": accepted / count if count else 0.0,
        "rollback_count": rollback,
        "rollback_rate": rollback / count if count else 0.0,
        "passthrough_count": passthrough,
        "passthrough_rate": passthrough / count if count else 0.0,
        "modified_count": modified,
        "modified_rate": modified / count if count else 0.0,
        "modified_and_accepted_count": modified_and_accepted,
        "modified_and_accepted_rate": modified_and_accepted / count if count else 0.0,
        "no_harm_count": no_harm,
        "no_harm_rate": no_harm / count if count else 0.0,
        "loadable_after_count": loadable_after,
        "loadable_after_rate": loadable_after / count if count else 0.0,
        "mean_observed_gain": sum(gains) / count if count else 0.0,
    }


def _read_strategies(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["strategy"]: row for row in csv.DictReader(handle)}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown(metrics: dict) -> str:
    strategies = metrics["strategies"]
    paired = metrics["paired_differences"]
    repair = metrics["repair_safe"]
    ready = metrics["ready3d_v2"]
    direct = strategies["direct"]
    ready2 = strategies["ready_top2"]
    all_s = strategies["all"]
    ready_vs_direct = paired["ready_top2_minus_direct_mean_quality"]
    lines = [
        "# ReadyRepair3D 最终汇报数据摘要",
        "",
        "## 主实验（Independent32）",
        "",
        "- 32个独立提示词组、每组4张SANA候选图、共128个Hunyuan3D结果。",
        (
            f"- Ready Top-2平均质量为{float(ready2['mean_quality_score_v2']):.4f}，"
            f"Direct为{float(direct['mean_quality_score_v2']):.4f}，"
            f"绝对提升{ready_vs_direct['estimate']:.4f}"
            f"（95% CI {ready_vs_direct['ci95_low']:.4f}～{ready_vs_direct['ci95_high']:.4f}）。"
        ),
        (
            f"- Ready Top-2合格率为{_pct(float(ready2['qualified_rate']))}，"
            f"Direct为{_pct(float(direct['qualified_rate']))}，"
            f"提高{(float(ready2['qualified_rate'])-float(direct['qualified_rate']))*100:.1f}个百分点。"
        ),
        (
            f"- All平均质量{float(all_s['mean_quality_score_v2']):.4f}、"
            f"合格率{_pct(float(all_s['qualified_rate']))}，平均调用4次3D生成。"
        ),
        (
            f"- Ready Top-2平均调用Hunyuan3D {float(ready2['mean_3d_calls']):.1f}次，"
            f"平均形状与纹理推理时间{float(ready2['mean_gpu_generation_seconds']):.1f}秒/组。"
        ),
        "",
        "## Ready3D V2",
        "",
        (
            f"- 训练/验证/测试：{ready['sample_counts']['train']}/"
            f"{ready['sample_counts']['validation']}/{ready['sample_counts']['test']}张，"
            f"对应{ready['group_counts']['train']}/{ready['group_counts']['validation']}/"
            f"{ready['group_counts']['test']}组。"
        ),
        f"- Ready Top-2质量捕获率为{_pct(metrics['selection_diagnostics']['ready_top2_quality_capture'])}。",
        "- 小样本测试集质量回归Spearman为负，说明Ready3D仍属于小规模可行性验证，泛化能力不能过度表述。",
        "",
        "## Repair3D安全主链路",
        "",
        (
            f"- 共检查{repair['report_count']}个最终资产：原资产直接通过{repair['passthrough_count']}个，"
            f"执行修改{repair['modified_count']}个。"
        ),
        (
            f"- 无伤率{_pct(repair['no_harm_rate'])}，"
            f"修复后仍可由Trimesh、Blender加载且数值有限的比例为{_pct(repair['loadable_after_rate'])}。"
        ),
        "- 正式演示使用确定性安全规则与复评分。",
        "",
        "## 一句话结论",
        "",
        "> 在32组独立实验中，Ready Top-2相对Direct显著提高平均自动质量与合格率；"
        "系统以完整可复现链路与安全修复收口，学习式筛选仍可继续扩大训练数据。",
        "",
    ]
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent32", type=Path, required=True)
    parser.add_argument("--ready-metrics", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    strategies_raw = _read_strategies(args.independent32 / "strategy_summary.csv")
    strategies = {
        name: {
            key: (float(value) if key not in {"strategy", "gpu_cost_scope"} else value)
            for key, value in row.items()
        }
        for name, row in strategies_raw.items()
    }
    confidence = json.loads(
        (args.independent32 / "bootstrap_confidence_intervals.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (args.independent32 / "selection_diagnostics.json").read_text(encoding="utf-8")
    )
    ready = json.loads(args.ready_metrics.read_text(encoding="utf-8"))
    repair_paths = sorted(args.repair_root.glob("*/repair_report.json"))
    metrics = {
        "evidence_scope": "frozen Independent32 plus Ready3D V2 and safe Repair3D reports",
        "strategies": strategies,
        "paired_differences": confidence["paired_differences"],
        "bootstrap": {key: confidence[key] for key in ("iterations", "seed", "group_count")},
        "selection_diagnostics": diagnostics,
        "ready3d_v2": {
            "sample_counts": ready["sample_counts"],
            "group_counts": ready["group_counts"],
            "quality": ready["quality"],
            "ranking": ready["ranking"],
        },
        "repair_safe": summarize_repairs(repair_paths),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "汇报数据摘要.md").write_text(
        build_markdown(metrics), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
