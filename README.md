# ReproFlow Agent

面向机器学习实验与论文证据管理的可复现 Agent 工作流系统。

> 当前里程碑：Day 1 已实现可复现的实验执行闭环；Agent 编排、RAG、报告和 Evidence
> Registry 的代码骨架已预留，但不属于 Day 1 验收范围。

## Day 1 已实现

- Apple Silicon / CPU 可运行的 sklearn 乳腺癌分类实验。
- Logistic Regression、Random Forest、SVM 三个 variant。
- 固定 seeds：42、43、44，共九次独立实验。
- 每次运行保存命令、环境、Git commit、脚本 SHA256、日志、指标和 manifest。
- 自动生成 `results.tsv`、`baseline.json` 和 `day1_summary.json`。
- 使用 `subprocess.run(..., shell=False)`，固定 120 秒 timeout。

核心思路参考 Andrej Karpathy 的
[autoresearch](https://github.com/karpathy/autoresearch)：以受控实验、固定评测和结构化结果记录
构成可审计的研究循环。本项目不是其 nanochat 训练代码的复制。

## 环境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- macOS / Linux / Windows CPU

```bash
# 在项目目录中
uv python install 3.12
uv sync --extra dev
```

## 运行 Day 1 实验矩阵

```bash
uv run python scripts/day1_run_all.py
```

也可以给运行指定唯一标签：

```bash
uv run python scripts/day1_run_all.py --tag first-demo
```

为保护历史结果，相同标签不会被覆盖，命令会直接失败并要求使用新标签。

输出结构：

```text
runs/<tag>/
├── logistic_regression-seed-42/
│   ├── artifacts/
│   ├── environment.json
│   ├── manifest.json
│   ├── metrics.json
│   ├── plan_snapshot.json
│   ├── stderr.log
│   └── stdout.log
├── ...其余八组实验
├── baseline.json
├── day1_summary.json
└── results.tsv
```

单独运行一组实验：

```bash
uv run python examples/sklearn_demo/experiment.py \
  --model logistic_regression \
  --seed 42 \
  --output /tmp/reproflow-metrics.json
```

## Day 1 验收

```bash
# 应返回 9/9 succeeded
uv run python scripts/day1_run_all.py --tag acceptance

# 验证固定指标文件存在
find runs/acceptance -name metrics.json | wc -l

# 开发检查（后续模块尚未纳入覆盖率门槛）
uv run ruff check examples scripts/day1_run_all.py
```

## 后续路线图

1. Pydantic 实验计划、Mock/API Planner 和人工审批。
2. 安全预检、超时/失败恢复与 LangGraph checkpoint。
3. JSON/CSV/Regex 解析、汇总图表和 Markdown 报告。
4. 工作记忆、历史实验记忆、Chroma RAG 和 ContextPack。
5. Evidence Registry、Streamlit UI、Agent evals 和端到端测试。

两分钟演示讲稿见 [`scripts/demo_2min.md`](scripts/demo_2min.md)。

## 许可证

MIT。见 [LICENSE](LICENSE)。

