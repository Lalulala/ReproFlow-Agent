# 仓库级 Agent 执行计划：本地合成二分类数据上的逻辑回归三随机种子 CPU 实验（修复依赖声明）

> 状态：`draft`
> 计划编号：`repo-plan-140393eb5f`
> 目标仓库：`$EVAL_ROOT/ml-from-scratch`

## 用户目标

选择仓库已有的分类模型和本地可用数据生成方式，设计三个随机种子的 CPU 快速实验，输出 JSON 指标。

修复目标：诊断计划 repo-plan-43ef7998e5 的失败并生成新的可审批方案。

## Agent 的判断

原始实验失败的根本原因是隔离虚拟环境中未安装仓库声明的 progressbar33 依赖。mlfromscratch.utils 包初始化时会从 misc.py 导入 progressbar，而 mlfromscratch.supervised_learning 包初始化又会导入 mlfromscratch.utils，因此 experiment_classification.py 在导入 LogisticRegression 时抛出 ModuleNotFoundError: No module named 'progressbar'。修复方式是在计划中显式声明 required_packages，让执行器使用隔离虚拟环境安装 progressbar33；同时保留三个随机种子（1、2、3）和同一个 variant（logistic_regression_make_classification），不修改全局 Python 环境，也不变更实验脚本或实验语义。

## 依赖与运行环境

- 模式：`isolated`
- Python：`3.12`
- 环境编号：`repo-env-e982778119ee`
- 判断：当前环境存在缺失或冲突依赖，批准后创建项目隔离的 uv 虚拟环境。

| 依赖 | 当前版本 | 状态 | 说明 |
| --- | --- | --- | --- |
| `cvxopt` | - | missing | not installed in the current ReproFlow environment |
| `matplotlib` | 3.11.1 | satisfied | installed 3.11.1 satisfies the requested requirement |
| `numpy` | 2.5.2 | satisfied | installed 2.5.2 satisfies the requested requirement |
| `pandas` | 3.0.5 | satisfied | installed 3.0.5 satisfies the requested requirement |
| `progressbar33` | - | missing | not installed in the current ReproFlow environment |
| `scikit-learn` | 1.9.0 | satisfied | installed 1.9.0 satisfies the requested requirement |
| `scipy` | 1.18.1 | satisfied | installed 1.18.1 satisfies the requested requirement |
| `terminaltables` | - | missing | not installed in the current ReproFlow environment |

批准后执行以下环境命令：

- `uv --cache-dir .reproflow/uv-cache venv --allow-existing --python 3.12 .reproflow/environments/repo-env-e982778119ee`
- `uv --cache-dir .reproflow/uv-cache pip install --python .reproflow/environments/repo-env-e982778119ee/bin/python cvxopt matplotlib numpy pandas progressbar33 scikit-learn scipy terminaltables`

## 修复来源

- 原失败计划：`repo-plan-43ef7998e5`
- 修复轮次：1/3

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
- `mlfromscratch/examples/experiment_classification.py`
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

Agent 判断无需修改代码，将直接复用仓库现有入口。
## 审批边界

审批前不会写入代码，也不会执行任何命令。
执行时不使用 Shell；隔离环境依赖仅按上方已批准命令安装，不允许 URL、本地路径、editable 依赖、内联 Python 和目录越界。
