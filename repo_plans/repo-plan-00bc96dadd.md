# 仓库级 Agent 执行计划：复现 sklearn 主实验：逻辑回归基线 vs 随机森林/SVM（42/43/44 种子）

> 状态：`completed`
> 计划编号：`repo-plan-00bc96dadd`
> 目标仓库：`$PROJECT_ROOT`

## 用户目标

找到仓库的主实验入口，复现 baseline，并用 42、43、44 三个随机种子比较新方法。

## Agent 的判断

仓库的主实验入口是 examples/sklearn_demo/experiment.py，当前代码已经支持三种模型并通过固定种子划分训练/测试集，因此不需要任何代码变更。本计划以 logistic_regression 为基线，将 random_forest 和 svm 作为待比较的新方法，在 42/43/44 三个随机种子下各运行一次，共 9 次独立实验。每次运行会写出 metrics.json（JSON 格式），包含 accuracy、f1、roc_auc、duration_seconds 等字段；ReproFlow 使用 parser=json 和 metrics_file 解析这些指标，并汇总生成 summary.csv/aggregate.csv，基线比较以 ROC-AUC 为主指标，同时记录 accuracy 和 F1。

## Agent 阅读了什么

- `README.md`
- `README_EN.md`
- `examples/sklearn_demo/experiment.py`
- `pyproject.toml`
- `scripts/day1_run_all.py`
- `src/reproflow/__init__.py`
- `src/reproflow/approval.py`
- `src/reproflow/cli.py`
- `src/reproflow/context.py`
- `src/reproflow/evidence.py`
- `src/reproflow/human_views.py`
- `src/reproflow/memory.py`
- `src/reproflow/metrics.py`
- `src/reproflow/models.py`
- `src/reproflow/planner.py`
- `src/reproflow/preflight.py`
- `src/reproflow/rag.py`
- `src/reproflow/repo_agent.py`
- `src/reproflow/reporting.py`
- `tests/test_day1.py`
- `tests/test_day2_planning.py`
- `tests/test_day3_day4.py`
- `tests/test_day5_day6.py`
- `tests/test_repo_agent.py`

## 拟执行实验

| Run | 用途 | Variant | Seed | 命令 | 超时 |
| --- | --- | --- | ---: | --- | ---: |
| `logistic_regression-seed-42` | 基线逻辑回归，种子 42 | logistic_regression | 42 | `python3 examples/sklearn_demo/experiment.py --model logistic_regression --seed 42 --output runs/compare_logistic_regression_seed42/metrics.json` | 120s |
| `logistic_regression-seed-43` | 基线逻辑回归，种子 43 | logistic_regression | 43 | `python3 examples/sklearn_demo/experiment.py --model logistic_regression --seed 43 --output runs/compare_logistic_regression_seed43/metrics.json` | 120s |
| `logistic_regression-seed-44` | 基线逻辑回归，种子 44 | logistic_regression | 44 | `python3 examples/sklearn_demo/experiment.py --model logistic_regression --seed 44 --output runs/compare_logistic_regression_seed44/metrics.json` | 120s |
| `random_forest-seed-42` | 新方法随机森林，种子 42 | random_forest | 42 | `python3 examples/sklearn_demo/experiment.py --model random_forest --seed 42 --output runs/compare_random_forest_seed42/metrics.json` | 120s |
| `random_forest-seed-43` | 新方法随机森林，种子 43 | random_forest | 43 | `python3 examples/sklearn_demo/experiment.py --model random_forest --seed 43 --output runs/compare_random_forest_seed43/metrics.json` | 120s |
| `random_forest-seed-44` | 新方法随机森林，种子 44 | random_forest | 44 | `python3 examples/sklearn_demo/experiment.py --model random_forest --seed 44 --output runs/compare_random_forest_seed44/metrics.json` | 120s |
| `svm-seed-42` | 新方法 SVM，种子 42 | svm | 42 | `python3 examples/sklearn_demo/experiment.py --model svm --seed 42 --output runs/compare_svm_seed42/metrics.json` | 120s |
| `svm-seed-43` | 新方法 SVM，种子 43 | svm | 43 | `python3 examples/sklearn_demo/experiment.py --model svm --seed 43 --output runs/compare_svm_seed43/metrics.json` | 120s |
| `svm-seed-44` | 新方法 SVM，种子 44 | svm | 44 | `python3 examples/sklearn_demo/experiment.py --model svm --seed 44 --output runs/compare_svm_seed44/metrics.json` | 120s |

## 拟修改代码

Agent 判断无需修改代码，将直接复用仓库现有入口。
## 审批边界

审批前不会写入代码，也不会执行任何命令。
执行时不使用 Shell，不自动安装依赖，不允许内联 Python 和目录越界。
