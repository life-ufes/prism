# PRISM: A Clinically Interpretable Stepwise Framework for Multimodal Skin Cancer Diagnosis


This repository contains the code and benchmarks for PRISM, focusing on multimodal skin cancer diagnosis using the PAD-UFES-20 and MILK10k datasets.

> **Before running anything:** update `config.py` to point to your local dataset copies.
>
> You must set the following paths correctly:
> - `PAD_20_PATH` and `PAD_20_IMAGES_FOLDER`
> - `MILK10K_PATH` and `MILK10K_TRAIN_IMAGES_FOLDER`

## 1. Preprocessing

Before running the experiments, you must generate the one-hot and sentence encodings for the metadata of the respective dataset.

```bash
# For PAD-UFES-20
python -m benchmarks.pad20.preprocess.onehot
python -m benchmarks.pad20.preprocess.sentence

# For MILK10k
python -m benchmarks.milk10k.preprocess.onehot
python -m benchmarks.milk10k.preprocess.sentence
```

## 2. Cross-Validation

You can run cross-validation on the feature fusion models using the `benchmarks.kfold` script. 

```bash
python -m benchmarks.kfold {pad20|milk10k} [OPTIONS]
```

**Options:**
- `--all`: Runs all baseline methods (`no_metadata`, `cross_attention`, `remixformer`, and `metablock`).
- `--fusion <method>`: Runs a single, specific fusion method (defaults to `no_metadata`).
- `--priors <timestamp>`: **Required** if running the `naive_bayes` model. You must provide the timestamp of the results folder from the `no_metadata` baseline.

*Note: Results are saved in the `benchmarks/{dataset}/results/{timestamp}` folder.*

## 3. Bayesian Network

The Bayesian Network approach is implemented distinctly using Pyro. 

First, run the Bayesian preprocessing script, passing the timestamp of the CNN `no_metadata` baseline results:
```bash
python -m benchmarks.{pad20|milk10k}.preprocess.bayesian --no-metadata-timestamp <timestamp>
```

Afterward, execute the cross-validation across all backbones:
```bash
python -m benchmarks.kfoldbayesian {pad20|milk10k}
```

## 4. Aggregating Results and Generating Tables

To generate LaTeX tables and aggregate all results, ensure that the output directories for all fusion methods (e.g., `cross_attention`, `naive_bayes`, `bayesiannetwork`, etc.) are placed under a single `benchmarks/{dataset}/results/{timestamp}` directory. 

Then run the metrics script:
```bash
python -m utils.metrics {pad20|milk10k} -t <timestamp>
```
This generates the LaTeX table with all metrics for all fusion methods and backbones.

## 5. Statistical Tests

After aggregating the results under the same `results/{timestamp}` directory, you can perform statistical tests:

```bash
python -m utils.statistical {pad20|milk10k} -t <timestamp> [--control <method>]
```

**Options:**
- `--control`: Specify the control method for comparisons (defaults to `naive_bayes`).

## 6. Incremental Evaluation on PAD-UFES-20

After aggregating the results for all methods under the same `results/{timestamp}` directory, you can run:

```bash
python -m benchmarks.pad20.incremental -t <timestamp>
```

**Options:**
- `--backbone`: Specifies the backbone (defaults to `efficientnet-b0`).
- `--cached`: Whether to use pre-computed results (defaults to False).

The new figures are saved to `benchmarks/pad20/results/{timestamp}/incremental_metadata_performance_subplots_*.png`

## Note

Recently, the sentence-transformers changed its API and started displaying the following warnings:

```bash
AlbertModel LOAD REPORT from: sentence-transformers/paraphrase-albert-small-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch."
```
You can safely ignore them: according to this [issue](https://github.com/huggingface/transformers/issues/44493),
the position_ids were "just an integer range tensor from 0 to the max sequence length. [...] there's not much point in saving this
in the checkpoints, since it can easily be recomputed on the fly", hence the error.
