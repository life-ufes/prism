from datetime import datetime
import argparse

from pathlib import Path
from benchmarks.benchmarks import Benchmarks, BenchmarksFactory
from sacred.observers import FileStorageObserver
from utils.metrics import (
    aggregate_results,
    generate_latex_macro_table,
    generate_latex_per_class_table,
)

if __name__ == "__main__":
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Run kfold validation")
    parser.add_argument(
        "benchmark",
        type=Benchmarks,
        choices=list(Benchmarks),
        help="Benchmark to perform cross-validation",
    )

    args = parser.parse_args()
    benchmark = args.benchmark

    experiment = BenchmarksFactory.get_bayesian_experiment(args.benchmark)
    models = [
        "davit_tiny",
        "efficientnet-b0",
        "mobilenet-v3",
        "swinv2_tiny",
    ]

    # define the base path in which the result is saved
    _results_dir = Path(f"benchmarks") / args.benchmark.value / "results"
    _version = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in models:
        _comb_path = _results_dir / _version / "bayesiannetwork" / model_name

        for folder in range(1, 6):
            save_folder = _comb_path / f"folder_{folder}"
            experiment.observers = []
            experiment.observers.append(FileStorageObserver.create(save_folder))
            config = {
                "save_folder": Path(save_folder),
                "folder": folder,
                "model_name": model_name,
                "early_stop_patience": 10,
                "batch_size": 65,
                "epochs": 100,
                "learning_rate": 2.5e-3,
                "early_stop_metric": "bacc",
            }
            experiment.run(config_updates=config)

    folds_df, overall_agg_df, per_class_agg_df = aggregate_results(
        _results_dir, timestamp_dir=_version, stage_filter="val", save=False
    )

    if not overall_agg_df.empty:
        generate_latex_macro_table(
            overall_agg_df,
            out_path=_results_dir / _version / "macro_metrics_summary.tex",
            caption="Aggregated macro metrics across all folds.",
            label="tab:agg_macro_metrics",
        )

    if not per_class_agg_df.empty:
        for metric in ["f1", "recall", "precision", "specificity", "auc"]:
            generate_latex_per_class_table(
                per_class_agg_df,
                out_dir=_results_dir / _version,
                metric=metric,
                caption_prefix=f'Per-class {metric.replace("_", " ").title()}',
            )
