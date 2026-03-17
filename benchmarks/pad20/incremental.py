"""
This script evaluates trained models on incrementally added metadata features.

It performs the following steps:
1.  Iterates through a specified timestamped results directory.
2.  For each combination of fusion method, CNN backbone, and fold, it loads a pre-trained model.
3.  It defines an incremental order of metadata features.
4.  For each model, it runs an evaluation loop, adding one metadata feature at a time.
5.  At each step, it calculates metrics (e.g., balanced accuracy) on the validation set.
6.  The script aggregates the results across all cross-validation folds.
7.  Finally, it generates and saves a plot showing metric performance as a function of the
    number of metadata features, with 95% confidence intervals for each fusion method.
"""

import os
from typing import List
import pyro
import torch
import argparse
import numpy as np
import pandas as pd
import lightning as L
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from scipy.stats import t
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from trainer.lightning import LightningAdapter

from models.factory import ClassifierFactory

from utils.metrics import compute_metrics
from utils.names import FEATURE_DISPLAY_NAMES, MODEL_DISPLAY_NAMES, METRIC_NAMES

from benchmarks.pad20.augmentation import ImgEvalTransform
from benchmarks.pad20.bayesian.model import HeMaskedBayesianNetwork
from benchmarks.pad20.bayesian.dataset import (
    PAD20 as PAD20_BAYESIAN,
    MaskedMetadataPAD20Bayesian,
)
from benchmarks.pad20.dataset import (
    MaskedMetadataPAD20,
    PAD20,
    MaskedMetadataPAD20SentenceEmbedding,
)

mpl.rcParams["font.family"] = "sans-serif"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

major, minor, revision = torch.__version__.split(".")
if int(major) < 2:
    raise ValueError("PyTorch version must be >= 2.0.0")
elif int(minor) >= 6:
    torch.serialization.add_safe_globals([torch.distributions.constraints._Simplex])


def run_incremental_evaluation(
    results_root: Path,
    timestamp: str,
    incremental_steps: List[str],
    backbone="efficientnet-b0",
):
    """
    Iterates through all methods, backbones (if not specified), and folds in the results directory.
    For each combination, it loads the corresponding model and dataset, then performs
    incremental evaluation by adding one metadata feature at a time and recording
    the metric at each step.

    At the end, it saves the aggregated results to a CSV file.
    """

    results_dir = results_root / timestamp
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    method_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    for method_dir in method_dirs:
        fusion_method_str = method_dir.name

        if "bayesian" not in fusion_method_str:
            continue

        backbone_dirs = [d for d in method_dir.iterdir() if d.is_dir()]
        for backbone_dir in backbone_dirs:
            model_name = backbone_dir.name
            if backbone is not None and model_name != backbone:
                continue
            fold_dirs = [
                d for d in backbone_dir.iterdir() if d.name.startswith("folder_")
            ]
            for fold_dir in fold_dirs:
                try:
                    fold = int(fold_dir.name.split("_")[-1])
                except Exception as e:
                    print(f"Could not parse fold number from {fold_dir.name}")
                    raise e

                # Handle 'no_metadata' and 'metablock-se' edge cases for fusion method
                fusion_method = (
                    None
                    if fusion_method_str == "no_metadata"
                    else fusion_method_str.replace("-se", "")
                )

                if fusion_method_str == "bayesiannetwork":
                    _run_incremental_evaluation_fold_bayesian(
                        fold_dir,
                        fold,
                        fusion_method_str,
                        model_name,
                        results_dir,
                        all_results,
                        incremental_steps,
                    )
                else:
                    _run_incremental_evaluation_fold(
                        results_dir,
                        fold_dir,
                        fold,
                        fusion_method_str,
                        model_name,
                        fusion_method,
                        device,
                        PAD20.METADATA_COLUMNS,
                        all_results,
                        incremental_steps,
                    )

    # Process and save results to CSV
    if not all_results:
        print("No results were generated. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    csv_path = results_dir / f"incremental_evaluation_raw_results_{backbone}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nIncremental evaluation complete. Results saved to {csv_path}")


def _run_incremental_evaluation_fold_bayesian(
    fold_dir,
    fold,
    fusion_method_str,
    model_name,
    results_dir,
    all_results,
    incremental_steps,
):
    checkpoint_path = fold_dir / "best_checkpoint" / "model.pt"
    if not checkpoint_path.exists():
        raise ValueError(
            f"Checkpoint not found for {results_dir}/{fusion_method_str}/{model_name}/fold_{fold}."
        )

    pyro.clear_param_store()
    pyro.get_param_store().load(checkpoint_path)
    model_instance = HeMaskedBayesianNetwork()

    metadata_path = fold_dir / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"metadata.csv not found in {fold_dir}.")

    df = pd.read_csv(metadata_path)
    val_dataset_full = PAD20_BAYESIAN(df, stage="val")

    # Incremental evaluation for Bayesian Network
    bayesian_features_raw = [
        f for f in PAD20_BAYESIAN.FEATURES if f not in PAD20.LABELS
    ]

    current_features = (
        PAD20.LABELS
    )  # we start with the vision probabilities, which are named as the labels
    for step_name in tqdm(
        incremental_steps,
        desc=f"Inc. features ({fusion_method_str} + {model_name} fold {fold})",
        leave=False,
    ):

        if step_name == "age":  # handle edge case
            current_features.append("age_group")
        elif (
            "diameter" in step_name and "diameter" not in current_features
        ):  # Avoid duplicated diameters because the bayesian network aggregates diameter_1 and diameter_2 using max().
            current_features.append("diameter")
        elif step_name in bayesian_features_raw:
            current_features.append(step_name)

        padded_dataset = MaskedMetadataPAD20Bayesian(
            val_dataset_full, features=current_features
        )
        val_dataloader = DataLoader(
            padded_dataset, batch_size=65, num_workers=4, shuffle=False
        )

        preds_df = validate_pyro(
            model_instance, val_dataloader, val_dataloader.dataset.to_label
        )

        metrics, _, _, _, _ = compute_metrics(preds_df)

        metrics_values = {
            "loss": np.zeros_like(metrics["accuracy"]),
            "acc": metrics["accuracy"],
            "f1": metrics["f1_macro"],
            "recall": metrics["balanced_accuracy"],
            "auc": metrics["auc_macro"],
        }

        all_results.append(
            {
                "fusion_method": fusion_method_str,
                "backbone": model_name,
                "fold": fold,
                "added_feature": step_name,
                "num_features": len(current_features),
                **metrics_values,
            }
        )

        # save the csv incrementally
        results_df = pd.DataFrame(all_results)
        csv_path = results_dir / f"incremental_evaluation_raw_results_{model_name}.csv"
        results_df.to_csv(csv_path, index=False)


@torch.no_grad()
def validate_pyro(model, dataloader, diagnostic_mapping):
    rows = []
    for img_ids, embeddings, labels in dataloader:
        embeddings, labels = embeddings.cuda(), labels.cuda()
        feature_args = embeddings.unbind(dim=1)
        _, probs = model.predict(*feature_args)
        for prob, img_id, label in zip(probs, img_ids, labels):
            row = {"id": img_id, "labels": dataloader.dataset.to_label(label.item())}
            for i, class_prob in enumerate(prob):
                row[diagnostic_mapping(i)] = class_prob.item()
            rows.append(row)

    return pd.DataFrame(rows)


def _run_incremental_evaluation_fold(
    results_dir,
    fold_dir,
    fold,
    fusion_method_str,
    model_name,
    fusion_method,
    device,
    all_meta_features,
    all_results,
    incremental_steps,
):
    is_sentence_embedding = fusion_method is not None and "metablock" in fusion_method

    n_total_features = len(all_meta_features) if not is_sentence_embedding else 768

    # Load Data for the fold from its specific metadata.csv
    metadata_path = fold_dir / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"metadata.csv not found in {fold_dir}.")

    df = pd.read_csv(metadata_path, index_col=0)
    val_mask = df["folder"] == fold
    val_df = df[val_mask]

    # Create the complete dataset
    val_dataset_full = PAD20(
        val_df,
        transforms=ImgEvalTransform(),
        meta_ordered_features=(  # keep the same order as in training
            PAD20.METADATA_COLUMNS if not is_sentence_embedding else []
        ),
    )

    # Load Model
    checkpoint_path = fold_dir / "best_checkpoint.pth"
    if not checkpoint_path.exists():
        raise ValueError(
            f"Checkpoint not found for {fusion_method_str}/{model_name}/fold_{fold}."
        )

    if "naive" not in fusion_method_str:
        model_instance = ClassifierFactory.get(
            n_classes=6,
            model_name=model_name,
            comb_method=fusion_method,
            n_metadata=n_total_features,
            checkpoint=checkpoint_path,
        )

        model_instance.eval()

    trainer = L.Trainer(logger=False, enable_progress_bar=False)

    # Incremental Evaluation Loop
    metrics = ["recall", "precision", "f1_macro", "specificity_macro", "auc_macro"]
    current_features = []
    old_metric_values = None
    steps = []
    for step_name in tqdm(
        incremental_steps,
        desc=f"Inc. features ({fusion_method_str} + {model_name} fold {fold})",
        leave=False,
    ):
        steps.append(step_name)
        if fusion_method is not None or old_metric_values is None:
            if step_name in PAD20.NUMERICAL_FEATURES:
                current_features.append(step_name)
            elif step_name in PAD20.RAW_CATEGORICAL_FEATURES:
                one_hot_cols = [
                    col
                    for col in PAD20.CATEGORICAL_FEATURES
                    if col.startswith(f"{step_name}_")
                ]
                current_features.extend(one_hot_cols)
            else:
                raise ValueError(
                    f"Step name {step_name} not found in either numerical or categorical features."
                )

            if fusion_method == "naive_bayes":
                # reload the model to update the feature indexes
                model_instance = ClassifierFactory.get(
                    n_classes=6,
                    model_name=model_name,
                    comb_method=fusion_method,
                    n_metadata=n_total_features,
                    checkpoint=checkpoint_path,
                    n_categorical_metadata=30,  # all categorical features one-hot encoded
                    n_numerical_metadata=1,
                )

                if not set(["age"]).intersection(set(current_features)):
                    model_instance.numerical_features_indexes = torch.empty(0)

                if not set(PAD20.CATEGORICAL_FEATURES).intersection(
                    set(current_features)
                ):
                    model_instance.categorical_features_indexes = torch.empty(0)

                model_instance.eval()

            lightning_model = LightningAdapter(
                model_instance, n_classes=len(PAD20.LABELS)
            ).to(device)

            feature_indices = (
                [all_meta_features.index(f) for f in current_features]
                if current_features
                else []
            )
            if fusion_method == "metablock":
                masked_dataset = MaskedMetadataPAD20SentenceEmbedding(
                    val_df,
                    SentenceTransformer(
                        "sentence-transformers/paraphrase-albert-small-v2"
                    ),
                    transforms=ImgEvalTransform(),
                    features=steps,
                )
            else:
                masked_dataset = MaskedMetadataPAD20(
                    val_dataset_full,
                    feature_indices=feature_indices,
                    total_features=n_total_features,
                    all_meta_features=all_meta_features,
                )

            val_dataloader = DataLoader(
                masked_dataset, batch_size=65, num_workers=16, shuffle=False
            )

            trainer.test(model=lightning_model, dataloaders=val_dataloader)
            metrics = lightning_model.metrics["_test"]

            metric_values = {
                metric_name: metrics[metric_name].compute().item()
                for metric_name in metrics
            }
        else:
            metric_values = old_metric_values

        all_results.append(
            {
                "fusion_method": fusion_method_str,
                "backbone": model_name,
                "fold": fold,
                "added_feature": step_name,
                "num_features": len(current_features),
                **metric_values,
            }
        )
        old_metric_values = metric_values

        # save the csv incrementally
        results_df = pd.DataFrame(all_results)
        csv_path = results_dir / f"incremental_evaluation_raw_results_{model_name}.csv"
        results_df.to_csv(csv_path, index=False)


def plot_results(
    results_root: Path,
    timestamp: str,
    metric_name="recall",
    backbone="efficientnet-b0",
):
    """
    Loads the evaluation results and generates plots.
    """

    results_dir = results_root / timestamp
    filename = (
        f"incremental_evaluation_raw_results_{backbone}.csv"
        if backbone
        else f"incremental_evaluation_raw_results.csv"
    )
    results_csv_path = results_dir / filename

    if not results_csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {results_csv_path}. Please run the evaluation script first."
        )

    results_df = pd.read_csv(results_csv_path)
    results_df.loc[:, "fusion_method"] = results_df.apply(
        lambda row: (
            row["fusion_method"]
            if row["fusion_method"] != "no_metadata"
            else row["backbone"]
        ),
        axis=1,
    )
    results_df.replace(FEATURE_DISPLAY_NAMES, inplace=True)
    results_df.replace(MODEL_DISPLAY_NAMES, inplace=True)

    backbones = results_df["backbone"].unique()
    n_backbones = len(backbones)

    # Create a flexible grid of subplots, aiming for 1 column
    n_cols = 1
    n_rows = (n_backbones + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    # Before plotting, manually aggregate and calculate CIs
    aggregated_data = []
    fusion_methods = [
        "Bayesian Network",
        "$\\mathbf{PRISM (ours)}$",
        "Cross Modality Fusion",
        "EfficientNet-B0",
        "MetaBlock-SE",
        "Cross-Attention",
    ]

    for backbone in results_df["backbone"].unique():
        for fusion_method in fusion_methods:
            backbone_fusion_df = results_df[
                (results_df["backbone"] == backbone)
                & (results_df["fusion_method"] == fusion_method)
            ]

            if backbone_fusion_df.empty:
                continue

            # For each feature, calculate mean and std and then calculate the confidence interval using the t-distribution
            for feature in backbone_fusion_df["added_feature"].unique():
                feature_data = backbone_fusion_df[
                    backbone_fusion_df["added_feature"] == feature
                ]
                mean_acc = feature_data[metric_name].mean()
                std_acc = feature_data[metric_name].std()

                n = len(feature_data)
                degrees_of_freedom = n - 1
                confidence_level = 0.95

                t_value = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

                aggregated_data.append(
                    {
                        "backbone": backbone,
                        "fusion_method": fusion_method,
                        "added_feature": feature,
                        metric_name: mean_acc,
                        "lower_ci": mean_acc - t_value * std_acc / np.sqrt(n),
                        "upper_ci": mean_acc + t_value * std_acc / np.sqrt(n),
                    }
                )

    # Create DataFrame from aggregated data
    agg_df = pd.DataFrame(aggregated_data)

    markers = [
        "D",
        "*",
        "s",
        None,
        "^",
        "o",
    ]
    colors = [
        "darkred",
        "tab:orange",
        "yellowgreen",
        "black",
        "tab:gray",
        "cornflowerblue",
    ]

    for i, backbone in enumerate(agg_df["backbone"].unique()):
        ax = axes[i]
        backbone_df = agg_df[agg_df["backbone"] == backbone]
        # Plot the line with your constant CI
        for j, fusion_method in enumerate(fusion_methods):
            highlight = "prism" in fusion_method.lower()
            method_df = backbone_df[backbone_df["fusion_method"] == fusion_method]

            if method_df.empty:
                continue
            ax.plot(
                method_df["added_feature"],
                method_df[metric_name],
                markersize=7 if not highlight else 16,
                markerfacecolor="none" if not highlight else colors[j],
                markeredgecolor=colors[j],
                markeredgewidth=2 if highlight else 1.5,
                linewidth=1 if not highlight else 2.5,
                marker=markers[j],
                label=fusion_method,
                color=colors[j],
            )
            ax.fill_between(
                method_df["added_feature"],
                method_df["lower_ci"],
                method_df["upper_ci"],
                alpha=0.2,
                color=colors[j],
            )
            ax.grid(True, linestyle="--", alpha=0.6)

        ax.set_ylabel(METRIC_NAMES.get(metric_name, metric_name), fontsize=16)
        ax.tick_params(axis="y", labelsize=16)

        if i == n_backbones - 1:
            ax.set_xlabel("Incrementally added feature", fontsize=16)
            ax.tick_params(axis="x", labelsize=16)
            plt.setp(
                ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor"
            )
            for label in ax.get_xticklabels():
                label.set_ha("right")
        else:
            # Hide x-axis labels for other subplots
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        ax.legend(
            title="Method",
            fontsize=16,
            title_fontsize=16,
            loc="best",
            bbox_to_anchor=(1.0, 1.02),
            edgecolor="black",
            framealpha=0.8,
        )

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plot_path = (
        results_dir
        / f"incremental_metadata_performance_subplots_{backbone}_{metric_name}.png"
        if backbone
        else results_dir
        / f"incremental_metadata_performance_subplots_{metric_name}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run incremental metadata evaluation.")
    parser.add_argument(
        "-t", type=str, help="Timestamp of the results directory to evaluate."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["efficientnet-b0", "mobilenet-v3", "davit_tiny", "swinv2_tiny"],
        default="efficientnet-b0",
        help="Backbone to evaluate (e.g., 'efficientnet-b0'). If not specified, evaluates all backbones.",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        default=False,
        help="Whether to use pre-computed results.",
    )
    args = parser.parse_args()

    incremental_steps = [
        "background_father",
        "background_mother",
        "skin_cancer_history",
        "pesticide",
        "diameter_1",
        "cancer_history",
        "gender",
        "fitspatrick",
        "diameter_2",
        "smoke",
        "drink",
        "has_sewage_system",
        "has_piped_water",
        "grew",
        "changed",
        "hurt",
        "itch",
        "bleed",
        "elevation",
        "region",
        "age",
    ]

    if not args.cached:
        run_incremental_evaluation(
            Path("benchmarks/pad20/results"),
            args.t,
            incremental_steps,
            backbone=args.backbone,
        )

    for metric in ["recall", "f1", "auc"]:
        plot_results(
            Path("benchmarks/pad20/results"),
            args.t,
            metric_name=metric,
            backbone=args.backbone,
        )
