import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from pathlib import Path
from utils.names import MODEL_DISPLAY_NAMES
from benchmarks.benchmarks import Benchmarks
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Dict, Tuple, Union, List, Optional, Sequence


def compute_metrics_from_csv(
    csv_path: Union[str, Path],
    stage_filter: str = "val",
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray, List[str]]:
    """
    Read a best_checkpoint_preds.csv file and compute multiclass metrics.

    Expected CSV columns:
    - labels: true class label names
    - id: sample identifier
    - stage: split (train/val/test)
    - <CLASS COLS...>: one column per class containing predicted probabilities

    Parameters
    ----------
    csv_path : str | Path
                    Path to the CSV file.
    stage_filter : str
                    Filter rows by stage value before computing metrics (e.g., "val").
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)

    required_cols = {"labels"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. Found: {set(df.columns)}"
        )

    # Stage filtering (e.g. val, test, etc.)
    if stage_filter is not None and "stage" in df.columns:
        df = df[df["stage"] == stage_filter].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(
                f"After stage filtering ({stage_filter}), no rows remain in {csv_path}"
            )

    # Infer class probability columns: anything that's not labels/id/stage
    non_class_cols = {"labels", "id", "stage"}
    class_cols: List[str] = [c for c in df.columns if c not in non_class_cols]

    # Heuristic validation for PAD-UFES and MILK10k datasets, which have 6 and 11 classes respectively.
    if len(class_cols) != 6 and len(class_cols) != 11:
        raise ValueError(
            "Could not infer class columns. Ensure the CSV has one probability column per class."
        )

    class_names = list(class_cols)
    n_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # True labels: map to indices since they are strings
    y_true_raw = df["labels"].values
    try:
        y_true = np.array([class_to_idx[str(lbl)] for lbl in y_true_raw])
    except KeyError as e:
        missing = sorted(set(map(str, y_true_raw)) - set(class_names))
        raise ValueError(
            f"Found label(s) not present in class columns: {missing}. Class columns: {class_names}"
        ) from e

    # Predicted probabilities and hard predictions
    y_prob = df[class_cols].to_numpy(dtype=float)
    if y_prob.shape[1] != n_classes:
        raise ValueError(
            f"Probability array has wrong number of columns: {y_prob.shape[1]} != n_classes={n_classes}"
        )
    y_pred = np.argmax(y_prob, axis=1)

    # Overall metrics (macro for multi-class where applicable)
    # Specificity is computed separately per class from the confusion matrix and then averaged (macro) since it's not directly available in sklearn for multi-class
    # AUC is computed with OVR strategy then macro averaged
    overall: Dict[str, float] = {}
    overall["accuracy"] = float(accuracy_score(y_true, y_pred))
    overall["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    overall["precision_macro"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    overall["recall_macro"] = float(recall_score(y_true, y_pred, average="macro"))
    overall["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # Per-class metrics
    labels_idx = np.arange(n_classes)
    per_prec = precision_score(
        y_true, y_pred, average=None, labels=labels_idx, zero_division=0
    )
    per_rec = recall_score(y_true, y_pred, average=None, labels=labels_idx)
    per_f1 = f1_score(y_true, y_pred, average=None, labels=labels_idx, zero_division=0)
    support = np.bincount(y_true, minlength=n_classes)

    # Specificity per class from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    per_spec = np.zeros(n_classes, dtype=float)
    total = cm.sum()
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        per_spec[i] = float(tn / denom) if denom > 0 else np.nan
    overall["specificity_macro"] = float(np.nanmean(per_spec))

    # AUC (macro and per class) with one-versus-rest (OVR)
    try:
        per_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average=None,
            labels=labels_idx,
        )
        overall["auc_macro"] = float(np.nanmean(per_auc))
    except Exception as e:
        print(f"Warning: Could not compute AUC for {csv_path}: {e}")
        per_auc = np.full(n_classes, np.nan, dtype=float)
        overall["auc_macro"] = float("nan")

    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "precision": per_prec,
            "recall": per_rec,
            "specificity": per_spec,
            "f1": per_f1,
            "auc": per_auc,
            "support": support,
        }
    ).set_index("class")

    return overall, per_class_df, y_true, y_pred, labels_idx, class_names


def aggregate_results(
    results_root: Union[str, Path],
    stage_filter: Optional[Union[str, Sequence[str]]] = None,
    csv_name: str = "best_checkpoint_preds.csv",
    save: bool = True,
    timestamp_dir: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Iterate a results directory structured as:
                    <results_root>/<timestamp>/<comb_method>/<cnn_backbone>/<folder_x>/<csv_name>

    Compute metrics per folder using compute_metrics_from_csv and aggregate across folds
    per (timestamp, comb_method, backbone).

    Returns three DataFrames:
                    - folds_df: one row per (timestamp, method, backbone, folder)
                    - overall_agg_df: aggregated macro metrics (mean, std) per (timestamp, method, backbone)
                    - per_class_agg_df: aggregated per-class metrics (mean, std) per (timestamp, method, backbone, class)

    If save=True, writes:
                    <results_root>/<timestamp>/<comb_method>/<cnn_backbone>/metrics_folds.csv
                    <results_root>/<timestamp>/<comb_method>/<cnn_backbone>/metrics_overall_agg.csv
                    <results_root>/<timestamp>/<comb_method>/<cnn_backbone>/metrics_per_class_agg.csv
    """

    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    fold_rows = []
    per_class_rows = []

    # Walk the tree: timestamp/comb/backbone/folder_x
    timestamp_dir = results_root / timestamp_dir
    for method_dir in sorted([p for p in timestamp_dir.iterdir() if p.is_dir()]):
        for backbone_dir in sorted([p for p in method_dir.iterdir() if p.is_dir()]):
            # Collect all fold dirs containing the csv
            fold_dirs = sorted([p for p in backbone_dir.iterdir() if p.is_dir()])

            # Check that we have the expected number of folds
            if len(fold_dirs) != 5:
                raise ValueError(
                    f"Expected 5 fold directories under {backbone_dir}, found {len(fold_dirs)}."
                )

            y_trues_all_folds = []
            y_preds_all_folds = []
            labels_idx = None
            class_names = None

            for fold_dir in fold_dirs:
                csv_path = fold_dir / csv_name

                if not csv_path.exists():
                    continue

                overall, per_class, y_true, y_pred, labels_idx, class_names = (
                    compute_metrics_from_csv(csv_path, stage_filter=stage_filter)
                )

                y_trues_all_folds.append(y_true)
                y_preds_all_folds.append(y_pred)

                if labels_idx is None:
                    labels_idx = labels_idx

                if class_names is None:
                    class_names = class_names
                elif class_names != class_names:
                    raise ValueError(f"Inconsistent class names in {fold_dir}.")

                # Fold-level macro metrics
                row = {
                    "timestamp": timestamp_dir.name,
                    "method": method_dir.name,
                    "backbone": backbone_dir.name,
                    "folder": fold_dir.name,
                    **overall,
                }
                fold_rows.append(row)

                # Per-class metrics rows
                for cls, vals in per_class.iterrows():
                    per_class_rows.append(
                        {
                            "timestamp": timestamp_dir.name,
                            "method": method_dir.name,
                            "backbone": backbone_dir.name,
                            "folder": fold_dir.name,
                            "class": cls,
                            **vals.to_dict(),
                        }
                    )

            # After iterating folds, if saving, write summaries for this (timestamp, method, backbone)
            if save:
                combo_fold_rows = [
                    r
                    for r in fold_rows
                    if r.get("timestamp") == timestamp_dir.name
                    and r.get("method") == method_dir.name
                    and r.get("backbone") == backbone_dir.name
                ]
                combo_per_class_rows = [
                    r
                    for r in per_class_rows
                    if r.get("timestamp") == timestamp_dir.name
                    and r.get("method") == method_dir.name
                    and r.get("backbone") == backbone_dir.name
                ]

                if combo_fold_rows:
                    folds_df_combo = pd.DataFrame(combo_fold_rows)
                    folds_df_combo.to_csv(
                        backbone_dir / "metrics_folds.csv", index=False
                    )

                    # Aggregate macro metrics per combo (mean, std)
                    metric_cols = [
                        c
                        for c in folds_df_combo.columns
                        if c not in ["timestamp", "method", "backbone", "folder"]
                    ]
                    agg_mean = folds_df_combo[metric_cols].mean(numeric_only=True)
                    agg_std = folds_df_combo[metric_cols].std(numeric_only=True)
                    overall_agg = pd.DataFrame(
                        {
                            "timestamp": [timestamp_dir.name],
                            "method": [method_dir.name],
                            "backbone": [backbone_dir.name],
                        }
                    )
                    for m in metric_cols:
                        overall_agg[f"{m}_mean"] = agg_mean.get(m, np.nan)
                        overall_agg[f"{m}_std"] = agg_std.get(m, np.nan)
                    overall_agg.to_csv(
                        backbone_dir / "metrics_overall_agg.csv", index=False
                    )

                if combo_per_class_rows:
                    per_class_df_combo = pd.DataFrame(combo_per_class_rows)

                    # Aggregate per-class metrics per combo (mean, std across folds)
                    metric_cols_pc = [
                        c
                        for c in [
                            "precision",
                            "recall",
                            "specificity",
                            "f1",
                            "auc",
                            "support",
                        ]
                        if c in per_class_df_combo.columns
                    ]
                    grouped = per_class_df_combo.groupby(["class"], as_index=False)
                    mean_df = grouped[metric_cols_pc].mean(numeric_only=True)
                    std_df = grouped[metric_cols_pc].std(numeric_only=True)

                    # Merge mean and std with suffixes
                    per_class_agg = mean_df.copy()
                    for m in metric_cols_pc:
                        per_class_agg.rename(columns={m: f"{m}_mean"}, inplace=True)
                        per_class_agg[f"{m}_std"] = std_df[m]

                    # Add identifiers
                    per_class_agg.insert(0, "backbone", backbone_dir.name)
                    per_class_agg.insert(0, "method", method_dir.name)
                    per_class_agg.insert(0, "timestamp", timestamp_dir.name)
                    per_class_agg.to_csv(
                        backbone_dir / "metrics_per_class_agg.csv", index=False
                    )

            # Plot and save aggregated confusion matrix ---
            if save and class_names:
                cm = confusion_matrix(
                    (
                        np.concatenate(y_trues_all_folds)
                        if y_trues_all_folds
                        else np.array([])
                    ),
                    (
                        np.concatenate(y_preds_all_folds)
                        if y_preds_all_folds
                        else np.array([])
                    ),
                    labels=labels_idx,
                    normalize="true",
                )

                plot_and_save_confusion_matrix(
                    cm,
                    class_names=class_names,
                    out_dir=backbone_dir,
                    display_names=MODEL_DISPLAY_NAMES,
                )

    # Build final DataFrames to return
    folds_df = pd.DataFrame(fold_rows)

    # overall aggregated: compute per (timestamp, method, backbone)
    if not folds_df.empty:
        id_cols = ["timestamp", "method", "backbone"]
        metric_cols = [c for c in folds_df.columns if c not in id_cols + ["folder"]]

        overall_agg_df = folds_df.groupby(id_cols)[metric_cols].agg(["mean", "std"])

        # flatten columns
        overall_agg_df.columns = [f"{m}_{stat}" for m, stat in overall_agg_df.columns]
        overall_agg_df = overall_agg_df.reset_index()
    else:
        overall_agg_df = pd.DataFrame()

    # per-class aggregated across folds
    per_class_df_all = pd.DataFrame(per_class_rows)
    if not per_class_df_all.empty:
        id_cols_pc = ["timestamp", "method", "backbone", "class"]
        metric_cols_pc = [
            c
            for c in ["precision", "recall", "specificity", "f1", "auc", "support"]
            if c in per_class_df_all.columns
        ]

        per_class_agg_df = per_class_df_all.groupby(id_cols_pc)[metric_cols_pc].agg(
            ["mean", "std"]
        )

        per_class_agg_df.columns = [
            f"{m}_{stat}" for m, stat in per_class_agg_df.columns
        ]
        per_class_agg_df = per_class_agg_df.reset_index()
    else:
        per_class_agg_df = pd.DataFrame()

    return folds_df, overall_agg_df, per_class_agg_df


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_dir: Union[str, Path],
    display_names: Optional[Dict[str, str]] = None,
    filename_prefix: str = "confusion_matrix_agg",
):
    """
    Plots and saves a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
                    The confusion matrix.
    class_names : List[str]
                    The names of the classes.
    out_dir : Union[str, Path]
                    The directory to save the plot and CSV.
    display_names : Optional[Dict[str, str]], optional
                    A mapping from internal class names to display names, by default None.
    filename_prefix : str, optional
                    The prefix for the output files, by default "confusion_matrix_agg".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw confusion matrix to a CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out_dir / f"{filename_prefix}.csv")

    fig, ax = plt.subplots(figsize=(10, 8))

    tick_labels = (
        [display_names.get(c, c) for c in class_names] if display_names else class_names
    )

    fontsize = (
        28 if len(class_names) <= 6 else 16
    )  # fontsize 28 for PAD-UFES-20 and 16 for MILK10k

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tick_labels)
    disp.plot(cmap="Oranges", xticks_rotation=45, ax=ax, text_kw={"fontsize": fontsize})
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    for text in disp.text_.ravel():
        text.set_text(f"{float(text.get_text()):.2f}")

    cbar = disp.im_.colorbar
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.ax.tick_params(labelsize=fontsize - 4)

    ax.set_xlabel("Predicted Label", fontsize=fontsize + 4)
    ax.set_ylabel("True Label", fontsize=fontsize + 4)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    plt.tight_layout()

    plot_path = out_dir / f"{filename_prefix}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Aggregated confusion matrix plot saved to {plot_path}")


# LaTeX table generation utilities
def _latex_escape(text: str) -> str:
    """Replace characters that have special meaning in LaTeX."""
    return str(text).replace("_", r"\_")


def _fmt_mean_std(
    mean_val: Optional[float], std_val: Optional[float], decimals: int = 3
) -> str:
    """Format mean and std-dev for LaTeX, e.g., '$0.123 \\pm 0.045$'."""
    if pd.isna(mean_val):
        return "-"

    std_val = 0.0 if pd.isna(std_val) else std_val

    mean_str = f"{mean_val:.{decimals}f}"
    std_str = f"{std_val:.{decimals}f}"

    return f"${mean_str}\\pm{std_str}$"


def generate_latex_macro_table(
    overall_agg_df: pd.DataFrame,
    out_path: Union[str, Path],
    timestamp: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    method_order: Optional[List[str]] = None,
    backbone_order: Optional[List[str]] = None,
    method_title_map: Optional[Dict[str, str]] = None,
    caption: str = "Performance summary of aggregated macro metrics.",
    label: str = "tab:macro_results",
) -> str:
    """
    Build a LaTeX table for aggregated macro metrics, grouped by comb_method ("method").
    """
    df = overall_agg_df.copy()
    if df.empty:
        print(
            f"Warning: No data found for the selected timestamp. Cannot generate table."
        )
        return ""

    if timestamp is not None:
        df = df[df["timestamp"] == timestamp]
    else:
        unique_ts = sorted(df["timestamp"].unique())
        if len(unique_ts) > 1:
            ts = unique_ts[-1]
            df = df[df["timestamp"] == ts]
            print(f"Multiple timestamps found. Using latest for table: {ts}")

    if df.empty:
        print(
            f"Warning: No data found for the selected timestamp. Cannot generate table."
        )
        return ""

    if metrics is None:
        metrics = [
            "balanced_accuracy",
            "f1_macro",
            "auc_macro",
            "specificity_macro",
            "precision_macro",
        ]

    if metric_labels is None:
        metric_labels = {
            "balanced_accuracy": "BACC",
            "f1_macro": "F1-Score",
            "auc_macro": "AUC",
            "specificity_macro": "Specificity",
            "precision_macro": "Precision",
        }

    if method_title_map is None:
        method_title_map = MODEL_DISPLAY_NAMES

    methods = [
        "no_metadata",
        "naive_bayes",
        "cross_attention",
        "remixformer",
        "bayesiannetwork",
        "metablock-se",
    ]
    backbones = (
        backbone_order
        if backbone_order is not None
        else sorted(list(df["backbone"].unique()))
    )

    ncols = 1 + len(metrics)
    lines: List[str] = []
    lines.append("\\begin{table}[htb]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{'l' + 'c' * len(metrics)}}}")
    lines.append("\\hline")

    for i, method in enumerate(methods):
        df_m = df[df["method"] == method].set_index("backbone")
        if df_m.empty:
            continue

        if i > 0:  # Add a hline between method blocks
            lines.append("\\hline")

        title = _latex_escape(method_title_map.get(method, method))
        lines.append(
            f"\shaderow \\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{title}}}}} \\\\"
        )

        header_cols = ["\\textbf{Model}"] + [
            f"\\textbf{{{_latex_escape(metric_labels.get(m, m))}}}" for m in metrics
        ]
        lines.append(" & ".join(header_cols) + " \\\\")
        lines.append("\\hline")

        for backbone in backbones:
            if backbone not in df_m.index:
                continue

            row = df_m.loc[backbone]
            cells = [
                f"\multicolumn{{1}}{{l|}}{{{_latex_escape(MODEL_DISPLAY_NAMES[backbone])}}}"
            ]
            for m in metrics:
                mean_col = f"{m}_mean"
                std_col = f"{m}_std"
                cells.append(_fmt_mean_std(row.get(mean_col), row.get(std_col)))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_str)
    print(f"LaTeX macro table saved to {out_path}")
    return latex_str


def generate_latex_per_class_table(
    per_class_agg_df: pd.DataFrame,
    out_dir: Union[str, Path],
    timestamp: Optional[str] = None,
    metric: str = "f1",
    method_order: Optional[List[str]] = None,
    backbone_order: Optional[List[str]] = None,
    class_order: Optional[List[str]] = None,
    method_title_map: Optional[Dict[str, str]] = None,
    caption_prefix: str = "Per-class performance",
    label_prefix: str = "tab:perclass",
) -> str:
    """
    Build a LaTeX table for a single aggregated per-class metric, grouped by method.
    """
    df = per_class_agg_df.copy()
    if df.empty:
        print("Warning: per_class_agg_df is empty. Cannot generate LaTeX table.")
        return ""

    # Data filtering
    if timestamp is not None:
        df = df[df["timestamp"] == timestamp]
    else:
        unique_ts = sorted(df["timestamp"].unique())
        if len(unique_ts) > 1:
            ts = unique_ts[-1]
            df = df[df["timestamp"] == ts]
            print(f"Multiple timestamps found. Using latest for table: {ts}")

    if df.empty:
        print(
            f"Warning: No data found for the selected timestamp. Cannot generate table."
        )
        return ""

    if method_title_map is None:
        method_title_map = MODEL_DISPLAY_NAMES

    methods = [
        "no_metadata",
        "naivebayes",
        "cross-attention",
        "remixformer",
        "bayesiannetwork",
        "metablock-se",
    ]
    backbones = (
        backbone_order
        if backbone_order is not None
        else sorted(list(df["backbone"].unique()))
    )
    classes = (
        class_order if class_order is not None else sorted(list(df["class"].unique()))
    )

    # LaTeX string building
    ncols = 1 + len(classes)
    lines: List[str] = []
    lines.append("\\begin{table}[htb]")
    lines.append(f"\\caption{{{caption_prefix} ({_latex_escape(metric)})}}")
    lines.append(f"\\label{{{label_prefix}:{metric}}}")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{'|l|' + 'c' * len(classes) + '|'}}}")
    lines.append("\\hline")

    for i, method in enumerate(methods):
        df_m = df[df["method"] == method]
        if df_m.empty:
            continue

        if i > 0:
            lines.append("\\hline")

        title = _latex_escape(method_title_map.get(method, method))
        lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{title}}}}} \\\\")
        lines.append("\\hline")

        header = ["\\textbf{Model}"] + [
            f"\\textbf{{{_latex_escape(c)}}}" for c in classes
        ]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\hline")

        for backbone in backbones:
            row_cells = [_latex_escape(backbone)]
            for cls in classes:
                row_df = df_m[(df_m["backbone"] == backbone) & (df_m["class"] == cls)]
                if row_df.empty:
                    row_cells.append("-")
                else:
                    r = row_df.iloc[0]
                    mean_val = r.get(f"{metric}_mean")
                    std_val = r.get(f"{metric}_std")
                    row_cells.append(_fmt_mean_std(mean_val, std_val))
            lines.append(" & ".join(row_cells) + " \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_str = "\n".join(lines)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"latex_per_class_{metric}.tex"
    out_file.write_text(latex_str)
    print(f"LaTeX per-class table for '{metric}' saved to {out_file}")
    return latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and aggregate metrics from benchmark results, then generate LaTeX tables."
    )
    parser.add_argument(
        "benchmark",
        type=Benchmarks,
        choices=list(Benchmarks),
        help="Benchmark to perform cross-validation",
    )
    parser.add_argument(
        "--stage-filter",
        type=str,
        default="val",
        help="Stage(s) to filter by (e.g., val, test, train)",
    )
    parser.add_argument(
        "-t", type=str, help="Restrict analysis to this timestamp subfolder only"
    )
    args = parser.parse_args()

    try:
        folds_df, overall_agg_df, per_class_agg_df = aggregate_results(
            f"benchmarks/{args.benchmark.value}/results",
            timestamp_dir=args.t,
            stage_filter=args.stage_filter,
            save=True,
        )

        # Generate laTeX tables
        latex_out_dir = Path(f"benchmarks/{args.benchmark.value}/results") / args.t

        if not overall_agg_df.empty:
            generate_latex_macro_table(
                overall_agg_df,
                out_path=latex_out_dir / "macro_metrics_summary.tex",
                caption="Aggregated macro metrics across all folds.",
                label="tab:agg_macro_metrics",
            )

        if not per_class_agg_df.empty:
            for metric in ["f1", "recall", "precision", "specificity", "auc"]:
                generate_latex_per_class_table(
                    per_class_agg_df,
                    out_dir=latex_out_dir,
                    metric=metric,
                    caption_prefix=f'Per-class {metric.replace("_", " ").title()}',
                )

    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure the results directory exists and is structured correctly."
        )
    except Exception as e:
        raise e
