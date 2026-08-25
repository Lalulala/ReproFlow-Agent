# ReproFlow Agent

面向机器学习实验与论文证据管理的可复现 Agent 工作流系统。

> 当前里程碑：Day 4 已完成。系统已走通“生成计划 → 人工审批 → 安全执行 →
> 失败恢复 → 指标解析 → CSV 汇总 → 图表”的工程闭环。

## 当前能力

### Day 1：可复现实验循环

- sklearn 乳腺癌分类 Demo：Logistic Regression、Random Forest、SVM。
- 固定 seeds 42、43、44，共九次独立 CPU 实验。
- 保存命令、环境、Git commit、脚本 SHA256、日志、指标和 manifest。

### Day 2：计划、安全预检与审批

- Pydantic `ExperimentPlan` 和可读的 YAML 计划。
- 无 Key 可用的 Mock Planner，以及 OpenAI-compatible API Planner。
- LLM 只允许改写标题和假设，不能控制命令、路径、seed、指标或 timeout。
- 拦截 Shell 操作符、目录越界、非白名单命令、参数注入和历史结果覆盖。
- 计划只有通过安全预检并由人工审批后才能执行，审批事件写入 SQLite。

### Day 3：Runner、失败处理与恢复

- `asyncio.create_subprocess_exec` 异步 Runner，始终以 argv 执行，不启用 Shell。
- 每组运行独立保存 YAML 快照、环境、stdout、stderr、metrics、manifest 和 artifacts。
- 支持 timeout、退出码、取消、部分失败和最多 1 MB 的日志限制。
- 子进程只继承白名单环境变量，环境快照不保存代理值或 API Key。
- LangGraph 状态图编排执行、解析、汇总和完成节点。
- SQLite checkpoint 位于 `.reproflow/checkpoints.sqlite`。
- `resume` 跳过已成功任务，只重试失败、超时或未完成任务；每次尝试均可审计。

### Day 4：解析、汇总与图表

- JSON、CSV、Regex 三种指标解析器。
- 缺失、非数值、NaN 或无穷指标不会被当作成功结果。
- `summary.csv`：逐次实验状态、耗时、指标和来源路径。
- `aggregate.csv`：按 variant 汇总 mean、sample std、best 和 baseline delta。
- `failures.csv`：失败、超时、取消和解析错误清单。
- `plots/metrics.png`：已验证指标的 mean ± std 图表。
- 所有汇总数字仅来自验证通过的指标文件。

核心实验循环参考 Andrej Karpathy 的
[autoresearch](https://github.com/karpathy/autoresearch)。ReproFlow 将其扩展为带审批、安全边界、
checkpoint 和证据溯源的通用科研工作流，不复制 nanochat 训练实现。

## 环境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- macOS / Linux / Windows CPU

```bash
uv python install 3.12
uv sync --extra dev
```

## 从目标到实验结果

初始化并创建计划：

```bash
uv run reproflow init .

REPROFLOW_RAG_BACKEND=lexical uv run reproflow plan \
  --goal "比较三类模型在三个随机种子下的效果" \
  --planner mock
```

命令会返回 `plan_id`。检查并人工审批：

```bash
uv run reproflow plan-show <plan_id>
uv run reproflow preflight <plan_id>
uv run reproflow approve <plan_id> \
  --actor Ethan \
  --reason "已核对实验矩阵、命令和输出路径"
```

执行批准后的九组实验：

```bash
uv run reproflow run <plan_id>
```

查看工作流状态和完整 Agent Trace：

```bash
uv run reproflow workflow-show <plan_id>
```

当前实现使用 `plan_id` 作为 `workflow_id`，从而保证同一批准计划不会覆盖或重复创建结果。

## 失败与恢复演示

在一个新的已审批计划上注入一次脚本错误和一次超时：

```bash
uv run reproflow run <plan_id> \
  --simulate-failure svm:43 \
  --simulate-timeout random_forest:44 \
  --timeout-seconds 1
```

该命令会生成部分成功的汇总并以非零状态退出。恢复时默认清除演示注入，仅重试失败任务：

```bash
uv run reproflow resume <plan_id>
```

模拟进程在完成两组实验后崩溃：

```bash
uv run reproflow run <plan_id> --crash-after 2
uv run reproflow resume <plan_id>
```

`--crash-after` 是一次性故障；checkpoint、SQLite 运行记录和磁盘 manifest 共同保证恢复。
需要重复注入故障时，可使用 `resume <plan_id> --keep-simulations`。执行期间按
`Control + C` 会取消当前子进程并记录 `cancelled` 状态。

## 输出结构

```text
runs/<workflow_id>/
├── <variant>-seed-<seed>/
│   ├── artifacts/
│   ├── plan_snapshot.yaml
│   ├── environment.json
│   ├── stdout.log
│   ├── stderr.log
│   ├── metrics.json
│   ├── manifest.json
│   └── attempts.jsonl       # 仅重试后出现，保留旧 attempt
├── summary.csv
├── aggregate.csv
├── failures.csv
└── plots/
    └── metrics.png
```

## OpenAI-compatible API

项目不会自动加载 `.env`。配置后先将其导入当前终端，再使用 API Planner：

```bash
set -a
source .env
set +a

REPROFLOW_RAG_BACKEND=lexical uv run reproflow plan \
  --goal "比较三类模型在三个随机种子下的效果" \
  --planner api
```

`.env` 已被 Git 忽略。API 输出仍必须通过 Schema，且不会获得可执行字段控制权。

## 测试与验收

```bash
uv run ruff check src tests
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests -q

REPROFLOW_RAG_BACKEND=lexical uv run pytest tests \
  --cov=reproflow.runner \
  --cov=reproflow.metrics \
  --cov=reproflow.workflow \
  --cov=reproflow.storage \
  --cov=reproflow.cli \
  --cov-report=term-missing -q
```

当前共 26 项测试通过；Day 3/4 相关核心模块组合覆盖率为 86%。测试覆盖正常执行、
审批门、crash、timeout、取消、缺指标、部分成功、幂等恢复、三种解析器、CSV 和图表。

真实 sklearn 验收结果：

- 正常工作流：9/9 成功。
- 故障工作流：先得到部分成功汇总，`resume` 后恢复到 9/9。
- 已成功任务保持 attempt 1，只有失败、超时或未完成任务增加 attempt。

## 后续路线图

1. Day 5：实验记忆、lessons、Chroma RAG 和分阶段 ContextPack。
2. Day 6：Markdown 报告、Evidence Registry、论文证据同步和 Streamlit UI。
3. Day 7：20 条 Agent eval、中英文演示材料和全流程复现。

两分钟演示讲稿见 [`scripts/demo_2min.md`](scripts/demo_2min.md)。

## 许可证

MIT。见 [LICENSE](LICENSE)。
