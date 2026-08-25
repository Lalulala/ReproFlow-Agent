# ReproFlow Day 4 两分钟演示讲稿

1. 输入实验目标，用 Mock 或 DeepSeek Planner 生成结构化 YAML。
2. 打开计划，指出模型、seed、指标、baseline、timeout 和受控 argv。
3. 运行 `preflight`，展示命令、路径、依赖和产物覆盖检查。
4. 由 Ethan 执行 `approve`，强调未经审批无法运行。
5. 执行 `reproflow run <plan_id>`，展示 LangGraph Trace 和九个独立目录。
6. 打开一个 `environment.json`、`metrics.json` 和 `manifest.json`，说明 commit、
   配置哈希、脚本哈希和指标溯源。
7. 打开 `summary.csv`、`aggregate.csv` 和 `plots/metrics.png`，指出 mean、std、
   best 和 baseline delta。
8. 切换到故障工作流：展示失败、超时和部分成功的 `failures.csv`。
9. 执行 `reproflow resume <workflow_id>`，展示成功任务 attempt 不变、仅失败任务重试。
10. 以 Day 5/6 路线图收尾：记忆/RAG、Markdown 报告和 Evidence Registry。
