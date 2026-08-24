# Bundled Model Notes

## Logistic Regression

Logistic regression is the baseline. Standardized features and a fixed random seed make it a strong,
interpretable reference for the binary classification task.

## Random Forest

The random-forest variant uses 120 trees and a bounded maximum depth. It represents a non-linear,
tree-based alternative.

## Support Vector Machine

The SVM variant uses standardized features, an RBF kernel, and probability calibration so ROC-AUC can
be calculated from predicted probabilities.

