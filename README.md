# ReproFlow Agent

面向机器学习实验与论文证据管理的可复现 Agent 工作流系统。

> 当前里程碑：Day 2 已完成。系统可以从自然语言目标生成结构化 YAML 实验计划，
> 执行安全预检，并通过带审计记录的人工审批门控制后续运行。

## 当前能力

### Day 1：可复现实验循环

- Apple Silicon / CPU 可运行的 sklearn 乳腺癌分类实验。
- Logistic Regression、Random Forest、SVM 三个 variant。
- 固定 seeds：42、43、44，共九次独立实验。
- 每次运行保存命令、环境、Git commit、脚本 SHA256、日志、指标和 manifest。
- 自动生成 `results.tsv`、`baseline.json` 和 `day1_summary.json`。
- 使用 `subprocess.run(..., shell=False)`，固定 120 秒 timeout。

### Day 2：计划、安全预检与审批

- Pydantic `ExperimentPlan`：目标、假设、受控 argv、variants、seeds、指标、baseline、路径和审批状态。
- Mock Planner：没有 API Key 也能稳定生成完整的 3 × 3 Demo 计划。
- OpenAI-compatible Planner：可优化标题与实验假设，但不能修改命令、路径等可执行字段。
- YAML 计划存储于 `.reproflow/plans/`，SQLite 保存计划版本和审批事件。
- 预检命令白名单、Python 解释器、脚本与数据路径、参数结构、依赖和产物冲突。
- 拦截 Shell 操作符、目录越界、非白名单命令、保留参数覆盖和历史结果覆盖。
- Git dirty 状态会提示警告，但不会掩盖真正的阻断项。
- 计划只有在预检全部安全后才能由人工批准；失败审批同样会留下审计记录。

核心思路参考 Andrej Karpathy 的
[autoresearch](https://github.com/karpathy/autoresearch)：以受控实验、固定评测和结构化结果记录
构成可审计的研究循环。本项目不是其 nanochat 训练代码的复制。

## 环境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- macOS / Linux / Windows CPU

```bash
uv python install 3.12
uv sync --extra dev
```

## Day 2 快速演示

初始化项目数据库和目录：

```bash
uv run reproflow init .
```

使用 Mock Planner 创建计划，不需要任何 API Key：

```bash
REPROFLOW_RAG_BACKEND=lexical uv run reproflow plan \
  --goal "比较三类模型在三个随机种子下的效果" \
  --planner mock
```

命令会返回 `plan_id`。查看、预检和审批该计划：

```bash
uv run reproflow plans
uv run reproflow plan-show <plan_id>
uv run reproflow preflight <plan_id>
uv run reproflow approve <plan_id> \
  --actor Ethan \
  --reason "已核对实验矩阵与输出路径"
```

拒绝不合适的草稿：

```bash
uv run reproflow reject <plan_id> --actor Ethan --reason "需要调整实验假设"
```

### OpenAI-compatible API 模式

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-provider.example/v1"
export REPROFLOW_MODEL="your-model"

REPROFLOW_RAG_BACKEND=lexical uv run reproflow plan \
  --goal "比较三类模型在三个随机种子下的效果" \
  --planner api
```

API 输出先通过 Pydantic Schema 校验，并且只采纳描述性字段；命令、variants、seeds、路径、
timeout 和指标定义均由本地安全模板锁定。API 不可用时可随时切回 `--planner mock`。

## 运行 Day 1 实验矩阵

```bash
uv run python scripts/day1_run_all.py --tag first-demo
```

为保护历史结果，相同标签不会被覆盖，命令会直接失败并要求使用新标签。单独运行一组实验：

```bash
uv run python examples/sklearn_demo/experiment.py \
  --model logistic_regression \
  --seed 42 \
  --output /tmp/reproflow-metrics.json
```

完整矩阵会写入 `runs/<tag>/`，每个模型与 seed 都有独立目录，并生成九组
`metrics.json`、日志、环境快照、manifest、baseline 和结果表。

## 测试与验收

```bash
uv run ruff check src tests
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests -q

# Day 2 核心模块覆盖率
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests/test_day2_planning.py \
  --cov=reproflow.models \
  --cov=reproflow.planner \
  --cov=reproflow.preflight \
  --cov=reproflow.approval \
  --cov=reproflow.storage \
  --cov=reproflow.cli \
  --cov-report=term-missing -q
```

当前 Day 2 核心模块覆盖率为 84%。API 模式通过模拟 OpenAI-compatible 响应测试其
Schema 和安全边界；真实服务需要用户自行提供 Key。

## 后续路线图

1. 异步 Runner、失败处理、幂等恢复和 LangGraph SQLite checkpoint。
2. JSON/CSV/Regex 解析、汇总图表和 Markdown 报告。
3. 工作记忆、历史实验记忆、Chroma RAG 和分阶段 ContextPack。
4. Evidence Registry、Streamlit UI、Agent evals 和端到端测试。

两分钟演示讲稿见 [`scripts/demo_2min.md`](scripts/demo_2min.md)。

## 许可证

MIT。见 [LICENSE](LICENSE)。
