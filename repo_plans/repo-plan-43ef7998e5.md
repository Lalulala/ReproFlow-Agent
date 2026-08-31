# 仓库级 Agent 执行计划：本地合成二分类数据上的逻辑回归三随机种子 CPU 实验

> 状态：`partial_failure`
> 计划编号：`repo-plan-43ef7998e5`
> 目标仓库：`$EVAL_ROOT/ml-from-scratch`

## 用户目标

选择仓库已有的分类模型和本地可用数据生成方式，设计三个随机种子的 CPU 快速实验，输出 JSON 指标。

## Agent 的判断

选择仓库已有的逻辑回归分类模型（mlfromscratch/supervised_learning/logistic_regression.py），新增一个轻量实验入口，使用 sklearn.datasets.make_classification 在本地生成可复现的二分类数据，避免依赖外部下载或需要显示绘图。三组实验只改变随机种子（1、2、3），分别输出 accuracy 的 JSON 指标，运行时间短，适合 CPU 快速验证。

## Agent 阅读了什么

- `README.md`
- `mlfromscratch/deep_learning/__init__.py`
- `mlfromscratch/deep_learning/activation_functions.py`
- `mlfromscratch/deep_learning/layers.py`
- `mlfromscratch/deep_learning/loss_functions.py`
- `mlfromscratch/deep_learning/neural_network.py`
- `mlfromscratch/deep_learning/optimizers.py`
- `mlfromscratch/examples/adaboost.py`
- `mlfromscratch/examples/apriori.py`
- `mlfromscratch/examples/bayesian_regression.py`
- `mlfromscratch/examples/convolutional_neural_network.py`
- `mlfromscratch/examples/dbscan.py`
- `mlfromscratch/examples/decision_tree_classifier.py`
- `mlfromscratch/examples/decision_tree_regressor.py`
- `mlfromscratch/examples/deep_q_network.py`
- `mlfromscratch/examples/demo.py`
- `mlfromscratch/examples/elastic_net.py`
- `mlfromscratch/examples/fp_growth.py`
- `mlfromscratch/examples/gaussian_mixture_model.py`
- `mlfromscratch/examples/genetic_algorithm.py`
- `mlfromscratch/examples/gradient_boosting_classifier.py`
- `mlfromscratch/examples/random_forest.py`
- `mlfromscratch/supervised_learning/__init__.py`
- `mlfromscratch/supervised_learning/decision_tree.py`
- `mlfromscratch/supervised_learning/random_forest.py`
- `mlfromscratch/utils/__init__.py`
- `mlfromscratch/utils/data_manipulation.py`
- `mlfromscratch/utils/data_operation.py`
- `mlfromscratch/utils/kernels.py`
- `mlfromscratch/utils/misc.py`
- `requirements.txt`
- `setup.py`

## 拟执行实验

| Run | 用途 | Variant | Seed | 命令 | 超时 |
| --- | --- | --- | ---: | --- | ---: |
| `lr-clf-seed-1` | 逻辑回归在本地合成二分类数据上的准确性（seed=1） | logistic_regression_make_classification | 1 | `python mlfromscratch/examples/experiment_classification.py --seed 1 --output metrics_seed_1.json` | 300s |
| `lr-clf-seed-2` | 逻辑回归在本地合成二分类数据上的准确性（seed=2） | logistic_regression_make_classification | 2 | `python mlfromscratch/examples/experiment_classification.py --seed 2 --output metrics_seed_2.json` | 300s |
| `lr-clf-seed-3` | 逻辑回归在本地合成二分类数据上的准确性（seed=3） | logistic_regression_make_classification | 3 | `python mlfromscratch/examples/experiment_classification.py --seed 3 --output metrics_seed_3.json` | 300s |

## 拟修改代码

### `mlfromscratch/examples/experiment_classification.py`

新增一个非绘图、可传种子的实验入口，复用仓库 LogisticRegression 模型，并输出 JSON 格式的 accuracy 指标；原示例均硬编码数据集且包含 plt.show()，不适合作为快速可复现的种子实验。

```diff
--- a/mlfromscratch/examples/experiment_classification.py
+++ b/mlfromscratch/examples/experiment_classification.py
@@ -0,0 +1,42 @@
+import argparse
+import json
+from sklearn import datasets
+from mlfromscratch.supervised_learning import LogisticRegression
+from mlfromscratch.utils import train_test_split, accuracy_score
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument('--seed', type=int, required=True)
+    parser.add_argument('--output', type=str, required=True)
+    args = parser.parse_args()
+
+    X, y = datasets.make_classification(
+        n_samples=500,
+        n_features=12,
+        n_informative=8,
+        n_redundant=2,
+        n_classes=2,
+        n_clusters_per_class=1,
+        flip_y=0.05,
+        random_state=args.seed,
+    )
+
+    X_train, X_test, y_train, y_test = train_test_split(
+        X, y, test_size=0.3, seed=args.seed
+    )
+
+    clf = LogisticRegression()
+    clf.fit(X_train, y_train)
+    y_pred = clf.predict(X_test)
+
+    accuracy = float(accuracy_score(y_test, y_pred))
+    result = {'seed': args.seed, 'accuracy': accuracy}
+
+    with open(args.output, 'w', encoding='utf-8') as f:
+        json.dump(result, f)
+    print(json.dumps(result))
+
+
+if __name__ == '__main__':
+    main()
```

## 审批边界

审批前不会写入代码，也不会执行任何命令。
执行时不使用 Shell，不自动安装依赖，不允许内联 Python 和目录越界。
