# 实验计划：比较三类模型在三个随机种子下的效果

> 当前状态：已完成
> 计划编号：`plan-8fb8de8fbb`

## 一、为什么做这个实验

**实验目标**

比较三类模型在三个随机种子下的效果

**待验证假设**

在三组固定的训练/测试划分上，至少有一种非线性模型的平均 ROC-AUC 会高于逻辑回归。

## 二、实验怎么做

本实验使用 **sklearn 内置乳腺癌分类数据集**，比较 逻辑回归（Logistic Regression）、随机森林（Random Forest）、支持向量机（SVM）。
逻辑回归（Logistic Regression） 作为基线模型。

每个模型分别使用随机种子 42、43、44 运行，总计 **9 组实验**。
单次实验最长允许运行 120 秒。

## 三、怎么判断结果

系统将记录 准确率（Accuracy）、F1 分数、ROC-AUC，并计算每个模型在多个随机种子下的平均值、
标准差、最佳值及相对基线的变化。

## 四、会产生什么

- 每次运行的日志、指标和环境快照。
- 逐次结果 `summary.csv` 和汇总结果 `aggregate.csv`。
- 指标图表和中文 Markdown 实验报告。
- 可供人工审核的论文 Evidence Claim。

## 五、审核与安全

已由 Ethan 审核批准。本次规划未检索到额外的本地知识或历史记忆。
系统只会执行白名单内的参数化命令，不使用 Shell，不会自动修改训练代码。

## 六、技术复现信息

- 实验脚本：`examples/sklearn_demo/experiment.py`
- 基础命令：`python3 examples/sklearn_demo/experiment.py`
- 产物根目录：`runs/`
- 机器可读计划：`.reproflow/plans/plan-8fb8de8fbb.yaml`
