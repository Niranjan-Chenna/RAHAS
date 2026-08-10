from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (2026, 17, 42, 123, 3407)
DATASET_SHA256 = "4d241e39f754b8cb4271eb94194eb07a706d50ccf61cf966063e87f91b0a8d7b"
METRICS = (
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("top3", "Top-3"),
    ("one_shot_accuracy", "One-shot"),
    ("2_4_accuracy", "2-4 samples"),
    ("5_9_accuracy", "5-9 samples"),
    ("10_plus_accuracy", "10+ samples"),
)
CONFIGURATIONS = (
    ("full_rapt", "Full RAHAS-RAPT"),
    ("resnet18", "Pretrained ResNet-18"),
    ("no_adaptive_routing", "RAPT without adaptive routing (fixed equal fusion)"),
    ("no_prototype_completion", "RAPT without prototype completion"),
    ("no_low_shot_specialist", "RAPT without low-shot specialist (direct only)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the focused RAHAS-RAPT ablation.")
    parser.add_argument(
        "--focused-root",
        type=Path,
        default=Path("pipeline/experiments/rahas_rapt_focused_ablation_v1"),
    )
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=Path("pipeline/experiments/rahas_rapt_validation_v1"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase1_values(path: Path) -> dict[str, float]:
    value = read_json(path)
    if value.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError(f"{path}: frozen dataset hash mismatch")
    return {key: float(value[f"test_{key}"]) for key, _ in METRICS}


def focused_values(path: Path) -> dict[str, float]:
    value = read_json(path)
    if value.get("status") != "PASS" or value.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError(f"{path}: evaluation integrity metadata is not PASS")
    test = value.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"{path}: missing test metrics")
    source_keys = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "top3": "top3",
        "one_shot_accuracy": "one_shot_accuracy",
        "2_4_accuracy": "low_shot_2_4_accuracy",
        "5_9_accuracy": "mid_shot_5_9_accuracy",
        "10_plus_accuracy": "many_shot_10_plus_accuracy",
    }
    return {key: float(test[source]) for key, source in source_keys.items()}


def load_results(focused_root: Path, phase1_root: Path) -> dict[str, dict[int, dict[str, float]]]:
    results = {key: {} for key, _ in CONFIGURATIONS}
    for seed in SEEDS:
        complete = read_json(focused_root / "runs" / f"seed_{seed}" / "COMPLETE.json")
        if complete.get("status") != "PASS" or complete.get("dataset_sha256") != DATASET_SHA256:
            raise ValueError(f"focused seed {seed} is not complete on the frozen dataset")
        results["full_rapt"][seed] = phase1_values(
            phase1_root / "seed_level_metrics" / f"rahasrapt_seed{seed}.json"
        )
        results["resnet18"][seed] = phase1_values(
            phase1_root / "seed_level_metrics" / f"resnet18_seed{seed}.json"
        )
        seed_root = focused_root / "runs" / f"seed_{seed}"
        results["no_adaptive_routing"][seed] = focused_values(
            seed_root / "no_adaptive_routing" / "metrics.json"
        )
        results["no_prototype_completion"][seed] = focused_values(
            seed_root / "no_prototype_completion_evaluation" / "metrics.json"
        )
        results["no_low_shot_specialist"][seed] = focused_values(
            seed_root / "no_low_shot_specialist" / "metrics.json"
        )
    return results


def aggregate_rows(results: dict) -> list[dict]:
    labels = dict(CONFIGURATIONS)
    rows = []
    for key, _ in CONFIGURATIONS:
        row = {"configuration_id": key, "configuration": labels[key], "seeds": len(SEEDS)}
        for metric, _ in METRICS:
            values = [results[key][seed][metric] for seed in SEEDS]
            mean = statistics.mean(values)
            sample_sd = statistics.stdev(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sample_sd"] = sample_sd
            row[f"{metric}_publication"] = f"{100 * mean:.2f} +/- {100 * sample_sd:.2f}"
        rows.append(row)
    return rows


def paired_rows(results: dict) -> list[dict]:
    labels = dict(CONFIGURATIONS)
    comparisons = (
        ("resnet18", "Full RAPT minus pretrained ResNet-18"),
        ("no_adaptive_routing", "Full RAPT minus no adaptive routing"),
        ("no_prototype_completion", "Full RAPT minus no prototype completion"),
        ("no_low_shot_specialist", "Full RAPT minus no low-shot specialist"),
    )
    rows = []
    for ablation, comparison in comparisons:
        for metric, _ in METRICS:
            differences = [
                results["full_rapt"][seed][metric] - results[ablation][seed][metric]
                for seed in SEEDS
            ]
            row = {
                "comparison": comparison,
                "ablation_id": ablation,
                "ablation": labels[ablation],
                "metric": metric,
                "mean_paired_difference": statistics.mean(differences),
                "sample_sd_paired_difference": statistics.stdev(differences),
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }
            for seed, difference in zip(SEEDS, differences):
                row[f"seed_{seed}_difference"] = difference
            rows.append(row)
    return rows


def paired_mean(rows: list[dict], ablation: str, metric: str) -> float:
    return float(next(
        row["mean_paired_difference"]
        for row in rows
        if row["ablation_id"] == ablation and row["metric"] == metric
    ))


def build_report(results: dict, table: list[dict], paired: list[dict]) -> str:
    resnet_consistent = all(
        results["full_rapt"][seed][metric] > results["resnet18"][seed][metric]
        for seed in SEEDS for metric in ("accuracy", "macro_f1")
    )
    routing_help = all(
        paired_mean(paired, "no_adaptive_routing", metric) > 0
        for metric in ("accuracy", "macro_f1")
    )
    completion_help = all(
        paired_mean(paired, "no_prototype_completion", metric) > 0
        for metric in ("one_shot_accuracy", "2_4_accuracy")
    )
    specialist_help = all(
        paired_mean(paired, "no_low_shot_specialist", metric) > 0
        for metric in ("one_shot_accuracy", "2_4_accuracy")
    )
    component_scores = {
        ablation: statistics.mean(
            paired_mean(paired, ablation, metric)
            for metric in ("accuracy", "macro_f1", "one_shot_accuracy", "2_4_accuracy", "5_9_accuracy")
        )
        for ablation in ("no_adaptive_routing", "no_prototype_completion", "no_low_shot_specialist")
    }
    strongest = max(component_scores, key=component_scores.get)
    strongest_name = {
        "no_adaptive_routing": "adaptive routing",
        "no_prototype_completion": "prototype completion",
        "no_low_shot_specialist": "the low-shot specialist",
    }[strongest]
    supported_components = sum((routing_help, completion_help, specialist_help))
    if resnet_consistent and supported_components == 3:
        verdict = "Core RAPT claim supported"
    elif resnet_consistent and supported_components >= 1:
        verdict = "Core RAPT claim partially supported"
    else:
        verdict = "Core RAPT claim unsupported"

    lines = [
        "# Focused RAHAS-RAPT Ablation Report",
        "",
        f"Frozen dataset SHA-256: `{DATASET_SHA256}`. Seeds: {', '.join(map(str, SEEDS))}. Values are test mean +/- sample SD in percentage points.",
        "",
        "## Publication-ready comparison",
        "",
        "| Configuration | Accuracy | Macro-F1 | Top-3 | One-shot | 2-4 samples | 5-9 samples | 10+ samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        cells = [row[f"{metric}_publication"] for metric, _ in METRICS]
        lines.append(f"| {row['configuration']} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Answers",
        "",
        f"1. **Does RAPT consistently outperform ResNet-18?** {'Yes' if resnet_consistent else 'No'}. Full RAPT beat ResNet-18 on both accuracy and macro-F1 for {sum(results['full_rapt'][s]['accuracy'] > results['resnet18'][s]['accuracy'] for s in SEEDS)}/5 and {sum(results['full_rapt'][s]['macro_f1'] > results['resnet18'][s]['macro_f1'] for s in SEEDS)}/5 seeds, respectively.",
        f"2. **Does learned routing contribute?** {'Yes' if routing_help else 'No consistent evidence'}. The implemented router is a validation-selected adaptive rule, not a learned neural router. Full-minus-fixed-fusion differences were {100 * paired_mean(paired, 'no_adaptive_routing', 'accuracy'):+.2f} accuracy and {100 * paired_mean(paired, 'no_adaptive_routing', 'macro_f1'):+.2f} macro-F1 points.",
        f"3. **Does prototype completion contribute to low-shot recognition?** {'Yes' if completion_help else 'No consistent evidence'}. Full-minus-no-completion differences were {100 * paired_mean(paired, 'no_prototype_completion', 'one_shot_accuracy'):+.2f} one-shot and {100 * paired_mean(paired, 'no_prototype_completion', '2_4_accuracy'):+.2f} points for 2-4 sample classes.",
        f"4. **Does the low-shot specialist improve one-shot and rare-class performance?** {'Yes' if specialist_help else 'No consistent evidence'}. Full-minus-direct-only differences were {100 * paired_mean(paired, 'no_low_shot_specialist', 'one_shot_accuracy'):+.2f} one-shot and {100 * paired_mean(paired, 'no_low_shot_specialist', '2_4_accuracy'):+.2f} points for 2-4 sample classes.",
        f"5. **Which component contributes most?** {strongest_name.capitalize()}, using the preregistered mean paired loss averaged across accuracy, macro-F1, one-shot, 2-4, and 5-9 sample accuracy ({100 * component_scores[strongest]:+.2f} points).",
        "",
        "## Final verdict",
        "",
        f"**{verdict}**",
        "",
        "The no-routing condition is fixed equal log-probability fusion. The no-specialist condition is direct-only inference from the same validation-selected full checkpoints. Prototype completion is the only retrained ablation and uses validation macro-F1 checkpoint selection before test access.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    focused_root = args.focused_root.resolve()
    results = load_results(focused_root, args.phase1_root.resolve())
    table = aggregate_rows(results)
    paired = paired_rows(results)
    write_csv(focused_root / "focused_ablation_table.csv", table)
    write_csv(focused_root / "focused_ablation_statistics.csv", paired)
    (focused_root / "FOCUSED_ABLATION_REPORT.md").write_text(
        build_report(results, table, paired), encoding="utf-8", newline="\n"
    )
    print(f"Focused ablation summary written to {focused_root}")


if __name__ == "__main__":
    main()
