"""Reproduce the additional studies locally, after both original task pipelines.

--resume skips only directories with completion markers. It is for continuing
the exact same checkout/data/config, not for reusing results after code changes.
"""
import argparse
import subprocess
import sys
from shared.data import ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true"); args = ap.parse_args()
    jobs = [("task1.temporal_study", "artifacts/task1_temporal/summary.csv"),
            ("task1.relational", "artifacts/task1_relational/runtime.json"),
            ("task1.relational_temporal", "artifacts/task1_relational_temporal/test_2021-08-09/runtime.json"),
            ("task1.ranking_study", "artifacts/task1_ranking/runtime.json"),
            ("task1.population_study", "artifacts/task1_population/runtime.json"),
            ("task1.cohort_study", "artifacts/task1_cohort/runtime.json"),
            ("task1.full_catalog", "artifacts/task1_full_catalog/runtime.json"),
            ("task1.catalog_calibration", "artifacts/task1_catalog_calibration/runtime.json"),
            ("task1.full_catalog --queries 512 --sampling-seed 20261027 --exclude-prior --calibration artifacts/task1_catalog_calibration/selection.json --out artifacts/task1_full_catalog_replication", "artifacts/task1_full_catalog_replication/runtime.json"),
            ("task1.rolling_catalog", "artifacts/task1_rolling_catalog/runtime.json"),
            ("task1.decayed_graph", "artifacts/task1_decayed_graph/runtime.json"),
            ("task2.robust_reward", "artifacts/task2_robust/runtime.json"),
            ("task2.analytic_policy", "artifacts/task2_analytic/runtime.json"),
            ("task2.history_mask", "artifacts/task2_masked/runtime.json"),
            ("task2.graph_warmstart", "artifacts/task2_warmstart/runtime.json"),
            ("task2.context_reward", "artifacts/task2_context_reward/runtime.json"),
            ("task2.context_rl", "artifacts/task2_context_rl/runtime.json"),
            ("task2.context_rl --warmstart-selected --out artifacts/task2_joint", "artifacts/task2_joint/runtime.json"),
            ("task2.reward_ablation", "artifacts/task2_reward_ablation/runtime.json"),
            ("task2.conditional_bandit", "artifacts/task2_conditional_bandit/runtime.json"),
            ("task2.return_reward", "artifacts/task2_return_reward/runtime.json"),
            ("task2.full_data_control", "artifacts/task2_full_data/runtime.json"),
            ("task2.conditional_bandit --full-reference --out artifacts/task2_conditional_full_reference", "artifacts/task2_conditional_full_reference/runtime.json"),
            ("task2.relevance_objectives", "artifacts/task2_relevance_objectives/audit.json"),
            ("task2.objective_validation", "artifacts/task2_objective_validation/runtime.json"),
            ("task2.validation_path", "artifacts/task2_validation_path/runtime.json"),
            ("shared.improvement_analysis", "task2/IMPROVEMENTS.md")]
    for module, marker in jobs:
        if args.resume and marker and (ROOT/marker).exists():
            print("SKIP completed", module, flush=True); continue
        print("RUN", module, flush=True)
        subprocess.run([sys.executable, "-m", *module.split()], cwd=ROOT, check=True)


if __name__ == "__main__": main()
