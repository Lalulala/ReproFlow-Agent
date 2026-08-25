# ReproFlow Agent

面向机器学习实验与论文证据管理的可复现 Agent 工作流系统。

> 当前里程碑：Day 6 已完成。系统已走通“科研目标 → RAG/历史记忆 → 结构化计划 →
> 人工审批 → 安全实验 → 失败恢复 → 指标汇总 → Markdown 报告 → Evidence 提议 →
> 人工审批同步”的闭环，并提供四页 Streamlit UI。

## Agent 能力

### 规划、安全与人机协同

- Mock Planner 无需 API Key；API Planner 支持 DeepSeek 等 OpenAI-compatible 服务。
- Pydantic 校验结构化计划，并保存为可读 YAML。
- LLM 只能改写标题和假设，不能控制命令、路径、seed、指标或 timeout。
- 预检拦截 Shell 操作符、目录越界、非白名单命令、参数注入和历史结果覆盖。
- 计划未经人工审批不能执行，Evidence 未经人工审批不能同步。

### 执行、恢复与可观测性

- LangGraph 编排执行、解析、汇总、分析、报告和证据提议节点。
- SQLite checkpoint 支持中断恢复；成功任务保持幂等，只重试失败或未完成任务。
- 异步、无 Shell 的 subprocess Runner，支持 timeout、取消、退出码和日志大小限制。
- 保存运行状态、Agent Trace、Git commit、配置哈希、脚本哈希及环境快照。

### Day 5：记忆、RAG 与上下文工程

- 工作记忆：LangGraph State 和 SQLite checkpoint。
- 情景记忆：自动沉淀实验摘要、失败历史和 dataset-specific lessons。
- RAG：索引本地 Markdown、TXT、PDF、论文证据文件和历史 `report.md`。
- Lexical 模式完全离线；Chroma 使用本地 `all-MiniLM-L6-v2` embedding。
- 检索结果包含来源路径、章节/页码、标签、分数和内容 SHA256。
- Planner 可读取知识与历史实验；Runner 只接收批准计划；Reporter 只接收验证数据。
- 无本地证据时明确返回 `No evidence found`。

### Day 6：报告、Evidence Registry 与 UI

- Jinja2 生成可复现 `report.md`，所有数字来自已验证的 CSV 和 `metrics.json`。
- Mock Narrator 可离线说明结果、局限和下一步；API Narrator 不允许生成数字。
- 自动建立 Claim—Experiment—Artifact 关系。
- Evidence 先进入 `proposed`，审核后成为
  `supported / contradicted / inconclusive`。
- 支持根据 Git commit 或配置哈希变化标记 `stale`。
- 只同步 `paper/evidence_registry.jsonl` 和 `paper/generated_results.md`，不改论文手稿。
- Streamlit 四页：Workflow、Runs、Evidence、Knowledge。

核心实验循环参考 Andrej Karpathy 的
[autoresearch](https://github.com/karpathy/autoresearch)。ReproFlow 将其扩展为带审批、记忆、
RAG、checkpoint 和论文证据溯源的科研 Agent，不复制 nanochat 训练实现。

## 安装

```bash
uv python install 3.12
uv sync --extra dev
```

要求 Python 3.12，支持 macOS、Linux 和 Windows CPU。

## 完整运行

创建计划：

```bash
uv run reproflow init .

REPROFLOW_RAG_BACKEND=lexical uv run reproflow plan \
  --goal "比较三类模型在三个随机种子下的效果" \
  --planner mock
```

查看 Planner 使用的最小 ContextPack：

```bash
REPROFLOW_RAG_BACKEND=lexical uv run reproflow context-show \
  --stage planner \
  --task "比较三类模型在三个随机种子下的效果"
```

检查并人工审批：

```bash
uv run reproflow plan-show <plan_id>
uv run reproflow preflight <plan_id>
uv run reproflow approve <plan_id> \
  --actor Ethan \
  --reason "已核对实验矩阵、命令和输出路径"
```

执行批准计划：

```bash
uv run reproflow run <plan_id>
```

运行结束会自动生成 CSV、图表、实验记忆、Markdown 报告和待审批 Evidence。查看状态：

```bash
uv run reproflow workflow-show <plan_id>
uv run reproflow memories
uv run reproflow evidence list
```

当前实现使用 `plan_id` 作为 `workflow_id`，保证同一批准计划不会覆盖历史结果。

## RAG

可靠的离线索引：

```bash
uv run reproflow knowledge index --backend lexical
uv run reproflow knowledge search "ROC-AUC protocol" --backend lexical
```

本地向量索引：

```bash
uv run reproflow knowledge index --backend chroma
uv run reproflow knowledge search "Which model performed best?" --backend chroma
```

Chroma 第一次运行会下载约 79 MB 的本地 MiniLM 模型。文档在本机完成 embedding，
向量数据库位于 `knowledge/.chroma/`。在向量索引尚未建立时，`auto` 自动使用 lexical，
避免规划命令意外触发模型下载。

## 报告

工作流默认使用 Mock Narrator 自动生成：

```text
runs/<workflow_id>/report.md
```

也可以在已加载 OpenAI-compatible 环境变量后重新生成 API 说明：

```bash
set -a
source .env
set +a

uv run reproflow report <workflow_id> --narrator api
```

API Reporter 会把已验证的 summary 和 aggregate 发送给所配置的模型服务。它被要求不输出
任何数字；若响应中出现数字，系统自动回退至 Mock Narrator。实验数值始终由模板直接读取。

## Evidence 审批与论文同步

查看 Claim 与完整溯源：

```bash
uv run reproflow evidence list
uv run reproflow evidence show <claim_id>
```

人工审核后批准：

```bash
uv run reproflow evidence approve <claim_id> --actor Ethan
```

只有存在已审核 Evidence 时才能同步：

```bash
uv run reproflow evidence sync
```

同步范围严格限制为：

```text
paper/evidence_registry.jsonl
paper/generated_results.md
```

检测代码或计划配置变化：

```bash
uv run reproflow evidence audit-stale <claim_id> --plan-id <plan_id>
```

## 失败与恢复

```bash
uv run reproflow run <plan_id> \
  --simulate-failure svm:43 \
  --simulate-timeout random_forest:44 \
  --timeout-seconds 1

uv run reproflow resume <plan_id>
```

`resume` 默认清除演示故障，并只重试失败任务。使用 `--crash-after 2` 可演示一次性
进程中断；执行期间按 Control+C 会记录 `cancelled`。

## Streamlit UI

```bash
uv run reproflow ui
```

- Workflow：输入目标、生成计划、人工审批、启动实验、阶段、报告和 Agent Trace。
- Runs：逐次指标、汇总表和图表。
- Evidence：Claim 查看、审批与同步。
- Knowledge：Lexical/Chroma 索引、带来源检索和实验记忆。

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
│   └── attempts.jsonl
├── summary.csv
├── aggregate.csv
├── failures.csv
├── report.md
└── plots/metrics.png
```

## 测试与验收

```bash
uv run ruff check src tests
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests -q
```

当前 34 项测试全部通过；Day 5/6 相关核心模块组合覆盖率为 81%。已验证：

- 第二次规划会引用第一次实验的 report、experiment memory 和 lesson。
- Chroma + 本地 MiniLM 实际索引 16 个知识块，并正确检索历史最佳模型。
- 9/9 成功工作流在恢复时跳过全部实验，只补生成记忆、报告和 Evidence。
- 报告中的数字和来源路径可回溯到 CSV、metrics 与 manifest。
- 未审批 Evidence 无法写入正式 Registry。
- 配置哈希变化后 Evidence 可标记 stale。
- Streamlit 四页均可执行，0 个页面异常。

## 后续路线图

Day 7：20 条 Agent eval、截图/GIF、两分钟视频、中英文发布材料和空目录复现。

两分钟演示讲稿见 [`scripts/demo_2min.md`](scripts/demo_2min.md)。

## 许可证

MIT。见 [LICENSE](LICENSE)。
