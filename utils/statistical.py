import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from pathlib import Path
from scipy.special import expit
from typing import Union, Optional, Sequence
from statsmodels.stats.multitest import multipletests

from benchmarks.benchmarks import Benchmarks
from utils.metrics import aggregate_results


def _run_lmm_pipeline(
    df: pd.DataFrame, metric: str, control_method: str, save_dir: Path
):
    """
    Runs the LMM pipeline for a single metric, including omnibus tests,
    estimated marginal means, and Holm-corrected post-hoc comparisons.
    """
    print(f"\n{'='*20} RUNNING LMM ANALYSIS ON '{metric.upper()}' {'='*20}")

    # Check if control method exists
    methods = df["method"].unique().tolist()
    if control_method not in methods:
        raise ValueError(
            f"Error: Control method '{control_method}' not found in the dataset."
        )

    baselines = [m for m in methods if m != control_method]

    # Apply Logit Transformation to map [0, 1] bounds to [-inf, inf] for the parametric LMM
    epsilon = 1e-5
    df["metric_clipped"] = np.clip(df[metric], epsilon, 1.0 - epsilon)
    df["metric_logit"] = np.log(df["metric_clipped"] / (1 - df["metric_clipped"]))

    # Setup crossed random effects using a dummy group
    df["dummy_group"] = 1

    # Force the control method to be the Reference Category for automatic contrast calculation
    df["method"] = pd.Categorical(df["method"], categories=[control_method] + baselines)

    print("Fitting Linear Mixed-Effects Model (LMM)...")
    formula = "metric_logit ~ C(method)"
    vc_formula = {"backbone": "0 + C(backbone)", "folder": "0 + C(folder)"}

    model = smf.mixedlm(
        formula, data=df, groups=df["dummy_group"], vc_formula=vc_formula
    )
    result = model.fit(method="lbfgs")

    print("\n--- Omnibus Test (Wald Test) ---")
    omnibus_test = result.wald_test_terms(scalar=False)
    omnibus_p_value = omnibus_test.table.loc["C(method)", "pvalue"]

    print(f"Global p-value for Fusion Methods: {omnibus_p_value:.6e}")

    if omnibus_p_value >= 0.05:
        print("\n[STOP] The Omnibus test is NOT significant.")
        print(
            "There is no statistically significant difference among the fusion methods."
        )
        print("Halting post-hoc comparisons to prevent Type I errors.")
        return result, None

    print(
        "[PASS] Significant differences detected among methods. Proceeding to post-hoc tests."
    )

    # Extract estimated marginal means (EMMs)
    control_logit_emm = result.params["Intercept"]
    control_emm_val = expit(control_logit_emm)

    baseline_emms = {}
    for baseline in baselines:
        param_name = f"C(method)[T.{baseline}]"
        baseline_coef = result.params[param_name]

        # Add the baseline's coefficient to the intercept to get its specific logit EMM
        baseline_logit_emm = control_logit_emm + baseline_coef
        baseline_emm_val = expit(baseline_logit_emm)
        baseline_emms[baseline] = baseline_emm_val

    print("\n--- Post-Hoc Pairwise Comparisons (Holm Corrected) ---")

    p_values = []
    comparisons = []

    # Extract unadjusted p-values from the model coefficients
    for baseline in baselines:
        param_name = f"C(method)[T.{baseline}]"
        p_values.append(result.pvalues[param_name])
        comparisons.append(baseline)

    # Apply the Holm step-down correction
    reject_null, pvals_corrected, _, _ = multipletests(
        p_values, alpha=0.05, method="holm"
    )

    # Compile the final results into a clean DataFrame
    results_df = pd.DataFrame(
        {
            "Control Method": [control_method] * len(baselines),
            "Control EMM": [control_emm_val] * len(baselines),
            "Baseline Method": comparisons,
            "Baseline EMM": [baseline_emms[b] for b in comparisons],
            "Difference": [control_emm_val - baseline_emms[b] for b in comparisons],
            "Raw p-value": p_values,
            "Holm Adjusted p-value": pvals_corrected,
            "Significant": reject_null,
        }
    )

    # Sort by significance for readability
    results_df = results_df.sort_values("Holm Adjusted p-value").reset_index(drop=True)

    return result, results_df


def run_statistical_tests(
    results_root: Union[str, Path],
    control_method: str,
    metrics: Sequence[str] = [
        "balanced_accuracy",
        "f1_macro",
        "auc_macro",
        "precision_macro",
        "specificity_macro",
    ],
    stage_filter: Optional[Union[str, Sequence[str]]] = "val",
    csv_name: str = "best_checkpoint_preds.csv",
    timestamp_dir: Optional[str] = None,
):
    """Loads data once and loops through all requested metrics."""

    # Build fold-level metrics
    folds_df, _ = aggregate_results(
        results_root=results_root,
        stage_filter=stage_filter,
        csv_name=csv_name,
        save=False,
        timestamp_dir=timestamp_dir,
    )

    if folds_df.empty:
        raise ValueError("No fold-level results found to run statistical tests.")

    # Determine save directory
    root = Path(results_root)
    save_dir = (root / timestamp_dir) if timestamp_dir else root
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        if metric not in folds_df.columns:
            print(f"Skipping {metric}: Not found in DataFrame columns.")
            continue

        # Create a fresh copy of the dataframe for this specific metric
        df_metric = folds_df.copy()

        # Filter out missing values for this metric
        df_metric = df_metric.dropna(subset=[metric])

        # Run Pipeline
        try:
            _, final_results_df = _run_lmm_pipeline(
                df=df_metric,
                metric=metric,
                control_method=control_method,
                save_dir=save_dir,
            )

            # Save CSV and LaTeX if omnibus test passed
            if final_results_df is not None:
                csv_path = save_dir / f"stats_lmm_{metric}.csv"
                tex_path = save_dir / f"stats_lmm_{metric}.tex"

                # Save CSV
                final_results_df.to_csv(csv_path, index=False)

                # Format for LaTeX
                latex_df = final_results_df.copy()
                latex_df.columns = [c.replace("_", "\\_") for c in latex_df.columns]

                try:
                    latex_str = latex_df.style.format(precision=4).to_latex(hrules=True)
                except AttributeError:
                    latex_str = latex_df.to_latex(
                        index=False, float_format="%.4f", escape=False
                    )

                with open(tex_path, "w") as f:
                    f.write(latex_str)

                print(
                    f"--> Saved reports for {metric} to:\n    {csv_path}\n    {tex_path}"
                )

        except Exception as e:
            print(f"Error processing metric {metric}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LMM statistical tests on benchmark results"
    )
    parser.add_argument(
        "benchmark",
        type=Benchmarks,
        choices=list(Benchmarks),
        help="Benchmark to perform cross-validation",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="naive_bayes",
        help="Control method to compare against (default: naive_bayes)",
    )
    parser.add_argument(
        "--stage-filter",
        type=str,
        default="val",
        help="Stage(s) to filter by (e.g., val, test, train)",
    )
    parser.add_argument(
        "-t", type=str, help="Restrict analysis to this timestamp results folder"
    )
    args = parser.parse_args()

    target_metrics = [
        "balanced_accuracy",
        "f1_macro",
        "auc_macro",
        "precision_macro",
        "specificity_macro",
    ]

    run_statistical_tests(
        results_root=f"benchmarks/{args.benchmark.value}/results",
        control_method=args.control,
        metrics=target_metrics,
        timestamp_dir=args.t,
        stage_filter=args.stage_filter,
    )
