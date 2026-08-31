# ReproFlow 多仓库兼容性评测

评测日期：2026-08-27
执行环境：Apple Silicon Mac、Python 3.12、CPU、DeepSeek OpenAI-compatible API

## 评测目标

验证 Repo Agent 面对不同代码结构时，能否完成：

1. 安全扫描公开 Git 仓库；
2. 自主选择需要阅读的代码；
3. 生成代码 Diff、实验矩阵、命令和指标规则；
4. 审批前保持零写入、零执行；
5. 审批后执行、记录失败并生成 CSV 与中文报告。

公开仓库均以 `--depth 1` 克隆到临时目录。计划生成后先人工审查 Diff，确认无网络、
删除、依赖安装或目录越界操作，再由 Ethan 批准执行。

## 结果矩阵

| 仓库 | 固定 commit | 安全扫描 | API 计划 | 运行结果 | 主要发现 |
| --- | --- | --- | --- | --- | --- |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | `7bc720e951fe` | 6 个文件 | 1 个新增脚本、3 个种子、2 项指标 | 3/3 成功 | 能复用本地微型神经网络实现；初版错误地把 seed 写入 Variant |
| [trekhleb/homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning) | `963d77f17f66` | 31 个文件 | 1 个新增脚本、3 个种子、4 项指标 | 0/3 成功 | 仓库旧 LogisticRegression 与当前 SciPy 不兼容，需二轮代码修复 |
| [eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) | `a2806c6732ee` | 85 个文件 | 1 个新增脚本、3 个种子、1 项指标 | 0/3 成功 | 修复 PYTHONPATH 后能找到本地包；随后发现目标仓库缺少 `progressbar` 依赖 |
| ReproFlow 内置 sklearn Demo | 本项目版本 | 自动识别入口 | 无需改代码、3 模型 × 3 种子 | 9/9 成功 | 完整产生 summary、aggregate、报告、记忆和 Evidence |
| API 生成的空仓库实验 Fixture | 测试固定输入 | README + 本地模块 | 在嵌套目录新增脚本 | 1/1 成功 | 验证审批前不落盘，以及嵌套脚本导入仓库顶层模块 |

## 可审计产物

### micrograd

- 计划：`repo_plans/repo-plan-1df23982a7.md`
- 报告：`runs/repo-plan-1df23982a7/report.md`
- 汇总：`runs/repo-plan-1df23982a7/aggregate.csv`

观测结果为 3/3 成功。由于初版计划将每个 seed 作为不同 Variant，CSV 中每组只有一个
样本，不能形成正确的跨种子标准差。这是计划语义错误，不是执行错误。

### homemade-machine-learning

- 计划：`repo_plans/repo-plan-8d6f861152.md`
- 报告：`runs/repo-plan-8d6f861152/report.md`
- 失败日志：`runs/repo-plan-8d6f861152/<run_id>/stderr.log`

三次运行一致失败于 SciPy：

```text
ValueError: 'x0' must only have one dimension.
```

这说明 Agent 找到了合理的模型入口，但没有先验证旧仓库代码与当前依赖版本的兼容性。
正确处理方式是生成新的修复 Diff 并再次审批，而不是静默修改原仓库。

### ML-From-Scratch

- 计划：`repo_plans/repo-plan-43ef7998e5.md`
- 报告：`runs/repo-plan-43ef7998e5/report.md`
- 失败日志：`runs/repo-plan-43ef7998e5/<run_id>/stderr.log`

首次运行因嵌套脚本无法导入仓库顶层包失败。Runner 增加受控 `PYTHONPATH` 后，该问题消失；
复测继续暴露目标仓库环境缺少 `progressbar`。该结果推动实现了 Dependency Preflight：新版
计划会把缺失依赖放入项目隔离的 uv 虚拟环境，且只有审批后才执行明确展示的安装 argv。

在获得用户明确的外部发送授权后，真实 DeepSeek Repair Agent 已基于失败日志生成 draft
`repo-plan-140393eb5f`。它保持原三种子矩阵与同一 Variant，不修改实验代码，提出在 Python
3.12 隔离环境中安装仓库执行路径所需的 `progressbar33`、`cvxopt`、`terminaltables` 等依赖。
该计划尚未批准或执行。

## 本轮已修复

- Runner 为目标进程设置只指向仓库根目录的 `PYTHONPATH`，支持嵌套实验入口导入本地包。
- Runner 设置 `MPLBACKEND=Agg`，避免 CPU/无界面实验意外打开绘图窗口。
- API Planner 被要求深读所有将要调用的仓库本地 API，而不是只根据文件名推测。
- API Planner 被要求 Variant 表示模型、方法或配置，并在不同随机种子之间保持一致。
- Guardrail 直接拒绝 `seed-<n>`、`seed=<n>` 一类错误 Variant。
- 单元测试覆盖“嵌套生成脚本导入仓库顶层模块”的真实运行路径。

## 当前结论

Repo Agent 已证明能够跨多个仓库完成安全探索、计划、代码生成、审批和可审计执行。本轮之后
已经加入 Dependency Preflight 和最多三轮的 Repair Agent：依赖冲突会规划隔离环境，运行或
环境失败会生成新的修复 Diff 并再次审批。

它仍不承诺“任何仓库一次成功”，也不会未经批准安装依赖或应用修复，因此应定位为能自主
诊断和提议修复的人机协同科研 Agent，而不是无人监督的自动代码修复器。
