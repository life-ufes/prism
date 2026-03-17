import os
import argparse

from pathlib import Path
from datetime import datetime
from sacred.observers import FileStorageObserver
from benchmarks.benchmarks import BenchmarksFactory, Benchmarks
from utils.statistical import run_statistical_tests
from utils.metrics import (
    aggregate_results,
    generate_latex_macro_table,
    generate_latex_per_class_table,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ALL_FEATURES_FUSIONS = ["cross_attention", "remixformer", "metablock", "naive_bayes"]

if __name__ == "__main__":
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Run kfold validation")
    parser.add_argument(
        "benchmark",
        type=Benchmarks,
        choices=list(Benchmarks),
        help="Benchmark to perform cross-validation",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=list(ALL_FEATURES_FUSIONS),
        required=False,
        help="Method to combine metadata features",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Train all methods: {['no_metadata'] + ALL_FEATURES_FUSIONS}",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=list(["one_hot", "sentence"]),
        default="one_hot",
        help="Method to used to preprocess the metadata",
    )
    parser.add_argument(
        "--priors",
        type=str,
        default=None,
        help="Timestamp of the experiment without metadata to use its priors to evaluate PRISM",
    )
    args = parser.parse_args()

    experiment = BenchmarksFactory.get_experiment(args.benchmark)

    _version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # configuration
    models = ["davit_tiny", "efficientnet-b0", "mobilenet-v3", "swinv2_tiny"]
    optimizer = "adam"
    best_metric = "loss"
    _preprocessing = args.preprocess

    # define the base path in which the result is saved
    _results_dir = Path(f"benchmarks") / args.benchmark.value / "results"
    _version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # no metadata checkpoints to be used for PRISM evaluation.
    # If --priors is provided, it uses the checkpoints from that timestamp,
    # otherwise it uses the checkpoints from the current timestamp
    no_metadata_timestamp = _version if args.all else args.priors

    no_metadata_checkpoints = (
        {
            model: {
                folder: Path(
                    f"benchmarks/{args.benchmark.value}/results/{args.priors}/no_metadata/{model}/folder_{folder}/best_checkpoint.pth"
                )
                for folder in range(1, 6)
            }
            for model in models
        }
        if args.priors is not None
        else None
    )

    feature_fusion_methods = (
        [None] + ALL_FEATURES_FUSIONS if args.all else [args.fusion]
    )
    for _comb_method in feature_fusion_methods:
        _preprocessing = "sentence" if _comb_method == "metablock" else "one_hot"

        # adds -se if using sentence embeddings
        metadata_comb_method = (
            f'{_comb_method}{"-se" if _preprocessing == "sentence" else ""}'
            if _comb_method
            else "no_metadata"
        )

        _comb_path = _results_dir / _version / metadata_comb_method
        for model_name in models:
            _model_path = _comb_path / model_name
            for folder in range(1, 6):
                # clear observers and create a new results folder
                _experiment_path = _model_path / f"folder_{folder}"
                experiment.observers = []
                experiment.observers.append(
                    FileStorageObserver.create(_experiment_path)
                )

                config = {
                    "_model_name": model_name,
                    "_batch_size": 65,
                    "_epochs": 100,
                    "_folder": folder,
                    "_comb_method": _comb_method,
                    "_preprocessing": _preprocessing,
                    "_early_stop_metric": "loss/val",
                    "_early_stop_patience": 10,
                    "_weight_by_frequency": True,
                    "_lr_initial": (1e-5 if _comb_method != "metablock" else 1e-4),
                    "_lr_scheduler_factor": 0.1,
                    "_lr_scheduler_patience": 5,
                    "_lr_scheduler_min_lr": (
                        1e-7 if _comb_method != "metablock" else 1e-6
                    ),
                    "_n_workers_train_dataloader": 8,
                    "_n_workers_val_dataloader": 4,
                    "_checkpoint_backbone": (
                        no_metadata_checkpoints[model_name][folder]
                        if no_metadata_checkpoints is not None
                        else None
                    ),
                    "_results_dir": _comb_path,
                    "_version": _experiment_path.stem,
                    "_experiment_path": _experiment_path,
                }
                experiment.run(config_updates=config)

    folds_df, overall_agg_df, per_class_agg_df = aggregate_results(
        _results_dir, timestamp_dir=_version, stage_filter="val", save=True
    )

    if args.all:
        run_statistical_tests(
            control_method="naive_bayes",
            results_root=_results_dir,
            timestamp_dir=_version,
            stage_filter="val",
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
