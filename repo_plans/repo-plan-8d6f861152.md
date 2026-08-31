# 仓库级 Agent 执行计划：基于仓库 LogisticRegression 的三种子合成数据分类实验

> 状态：`partial_failure`
> 计划编号：`repo-plan-8d6f861152`
> 目标仓库：`$EVAL_ROOT/homemade-machine-learning`

## 用户目标

选择仓库中合适的分类算法，在无需下载数据的前提下设计三个随机种子的可复现实验，输出 JSON 指标。

## Agent 的判断

仓库中合适的分类算法是自制 LogisticRegression。由于没有可直接运行的现成实验入口且仓库内没有可用的 CSV 数据文件（不可下载数据），因此新增一个基于 numpy 生成多类高斯 blob 数据的脚本：在同一随机种子下完成数据生成、训练/测试划分，训练 One-vs-All 多分类 LogisticRegression，并将准确率、宏平均精确率、宏平均召回率、宏平均 F1 写入 JSON。设计 0/1/2 三个随机种子，保证实验可复现。

## Agent 阅读了什么

- `README.es-ES.md`
- `README.md`
- `data/README.md`
- `homemade/anomaly_detection/README.md`
- `homemade/anomaly_detection/__init__.py`
- `homemade/anomaly_detection/gaussian_anomaly_detection.py`
- `homemade/k_means/README.md`
- `homemade/k_means/__init__.py`
- `homemade/k_means/k_means.py`
- `homemade/linear_regression/README.md`
- `homemade/linear_regression/__init__.py`
- `homemade/linear_regression/linear_regression.py`
- `homemade/logistic_regression/README.md`
- `homemade/logistic_regression/__init__.py`
- `homemade/logistic_regression/logistic_regression.py`
- `homemade/neural_network/README.md`
- `homemade/neural_network/__init__.py`
- `homemade/neural_network/multilayer_perceptron.py`
- `homemade/utils/features/__init__.py`
- `homemade/utils/features/generate_polynomials.py`
- `homemade/utils/features/generate_sinusoids.py`
- `homemade/utils/features/prepare_for_training.py`
- `requirements.txt`

## 拟执行实验

| Run | 用途 | Variant | Seed | 命令 | 超时 |
| --- | --- | --- | ---: | --- | ---: |
| `logistic-regression-seed-0` | 使用随机种子 0 生成数据、划分训练/测试集并训练 LogisticRegression，输出指标 JSON。 | seed-0 | 0 | `python3 logistic_regression_blobs_experiment.py --seed 0 --metrics-file metrics_seed_0.json --num-examples 300 --num-features 4 --num-classes 3 --test-fraction 0.3 --lambda-param 0.01 --max-iterations 300` | 600s |
| `logistic-regression-seed-1` | 使用随机种子 1 生成数据、划分训练/测试集并训练 LogisticRegression，输出指标 JSON。 | seed-1 | 1 | `python3 logistic_regression_blobs_experiment.py --seed 1 --metrics-file metrics_seed_1.json --num-examples 300 --num-features 4 --num-classes 3 --test-fraction 0.3 --lambda-param 0.01 --max-iterations 300` | 600s |
| `logistic-regression-seed-2` | 使用随机种子 2 生成数据、划分训练/测试集并训练 LogisticRegression，输出指标 JSON。 | seed-2 | 2 | `python3 logistic_regression_blobs_experiment.py --seed 2 --metrics-file metrics_seed_2.json --num-examples 300 --num-features 4 --num-classes 3 --test-fraction 0.3 --lambda-param 0.01 --max-iterations 300` | 600s |

## 拟修改代码

### `logistic_regression_blobs_experiment.py`

新增可复现实验脚本：使用 numpy 生成多类高斯 blobs，避免下载外部数据；调用 homemade.logistic_regression.LogisticRegression 完成多分类训练与预测；将准确率、宏平均精确率/召回率/F1 写入 JSON 指标文件。

```diff
--- a/logistic_regression_blobs_experiment.py
+++ b/logistic_regression_blobs_experiment.py
@@ -0,0 +1,122 @@
+#!/usr/bin/env python3
+"""Reproducible logistic-regression experiment on synthetic blob data.
+
+The experiment does not download any dataset. It creates a synthetic multi-class
+dataset by sampling Gaussian blobs, splits it with a fixed seed, trains the
+repository's homemade LogisticRegression and writes a metrics JSON.
+"""
+
+import argparse
+import json
+
+import numpy as np
+
+from homemade.logistic_regression import LogisticRegression
+
+
+def make_blobs(seed, num_examples=300, num_features=4, num_classes=3):
+    """Create a deterministic synthetic multi-class dataset."""
+    rng = np.random.RandomState(seed)
+    centers = rng.normal(scale=3.0, size=(num_classes, num_features))
+    per_class = num_examples // num_classes
+    examples = []
+    labels = []
+    for class_id in range(num_classes):
+        class_data = centers[class_id] + rng.normal(
+            scale=0.8, size=(per_class, num_features)
+        )
+        examples.append(class_data)
+        labels.append(np.full((per_class, 1), class_id, dtype=int))
+    data = np.vstack(examples)
+    target = np.vstack(labels)
+    order = rng.permutation(num_examples)
+    return data[order], target[order]
+
+
+def train_test_split(data, labels, test_fraction, seed):
+    """Split data with a fixed seed."""
+    rng = np.random.RandomState(seed)
+    order = rng.permutation(data.shape[0])
+    test_size = int(data.shape[0] * test_fraction)
+    test_idx = order[:test_size]
+    train_idx = order[test_size:]
+    return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]
+
+
+def macro_metrics(true_labels, predicted_labels, num_classes):
+    """Compute accuracy plus macro-averaged precision/recall/F1."""
+    confusion = np.zeros((num_classes, num_classes), dtype=int)
+    for true_label, pred_label in zip(true_labels, predicted_labels):
+        confusion[int(true_label), int(pred_label)] += 1
+
+    def safe_div(numerator, denominator):
+        return float(numerator / denominator) if denominator != 0 else 0.0
+
+    accuracy = safe_div(np.trace(confusion), np.sum(confusion))
+    precisions = []
+    recalls = []
+    f1_scores = []
+    for class_id in range(num_classes):
+        true_positive = confusion[class_id, class_id]
+        false_positive = np.sum(confusion[:, class_id]) - true_positive
+        false_negative = np.sum(confusion[class_id, :]) - true_positive
+        precision = safe_div(true_positive, true_positive + false_positive)
+        recall = safe_div(true_positive, true_positive + false_negative)
+        f1 = safe_div(2 * precision * recall, precision + recall)
+        precisions.append(precision)
+        recalls.append(recall)
+        f1_scores.append(f1)
+    return {
+        "accuracy": accuracy,
+        "precision_macro": float(np.mean(precisions)),
+        "recall_macro": float(np.mean(recalls)),
+        "f1_macro": float(np.mean(f1_scores)),
+    }
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--seed", type=int, required=True)
+    parser.add_argument("--metrics-file", required=True)
+    parser.add_argument("--num-examples", type=int, default=300)
+    parser.add_argument("--num-features", type=int, default=4)
+    parser.add_argument("--num-classes", type=int, default=3)
+    parser.add_argument("--test-fraction", type=float, default=0.3)
+    parser.add_argument("--lambda-param", type=float, default=0.01)
+    parser.add_argument("--max-iterations", type=int, default=300)
+    args = parser.parse_args()
+
+    data, labels = make_blobs(
+        seed=args.seed,
+        num_examples=args.num_examples,
+        num_features=args.num_features,
+        num_classes=args.num_classes,
+    )
+    x_train, y_train, x_test, y_test = train_test_split(
+        data, labels, args.test_fraction, seed=args.seed
+    )
+
+    model = LogisticRegression(
+        x_train,
+        y_train,
+        polynomial_degree=0,
+        sinusoid_degree=0,
+        normalize_data=True,
+    )
+    model.train(lambda_param=args.lambda_param, max_iterations=args.max_iterations)
+    predictions = model.predict(x_test).flatten()
+    predicted_labels = np.asarray([int(label) for label in predictions])
+
+    true_labels = y_test.flatten().astype(int)
+    metrics = macro_metrics(true_labels, predicted_labels, args.num_classes)
+    metrics["seed"] = args.seed
+    metrics["num_train"] = int(x_train.shape[0])
+    metrics["num_test"] = int(x_test.shape[0])
+
+    with open(args.metrics_file, "w", encoding="utf-8") as metrics_file:
+        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
+    print(json.dumps(metrics, sort_keys=True))
+
+
+if __name__ == "__main__":
+    main()
```

## 审批边界

审批前不会写入代码，也不会执行任何命令。
执行时不使用 Shell，不自动安装依赖，不允许内联 Python 和目录越界。
