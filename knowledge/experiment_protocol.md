# ReproFlow Experiment Protocol

## Goal

All experiments must be reproducible, bounded, and traceable to an immutable configuration snapshot.

## Baseline

The bundled demonstration uses logistic regression as the baseline. Random forest and support vector
machine variants are evaluated on the same breast-cancer dataset and the same three split seeds.

## Metrics

The primary metric is ROC-AUC. Accuracy and F1 are secondary metrics. A result is not a paper claim
until every source run succeeds and the evidence record links to the metric files.

## Reproducibility

Every run stores its command, Python environment, platform, Git commit, script hash, stdout, stderr,
metrics, and manifest in an independent run directory.

