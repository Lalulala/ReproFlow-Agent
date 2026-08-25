# ReproFlow Day 6 两分钟演示讲稿

1. 输入实验目标，展示 Planner ContextPack 中的本地协议、历史 report 和 lesson。
2. 打开生成的 YAML，指出 LLM 不能控制命令、路径、seed 与指标。
3. 运行 `preflight` 并由 Ethan 审批计划。
4. 执行 `reproflow run <plan_id>`，展示 LangGraph Trace 和九个独立运行目录。
5. 打开 `summary.csv`、`aggregate.csv` 与图表，指出 mean、std、best 和 baseline delta。
6. 打开 `report.md`，说明表格数字直接来自验证文件，Narrator 只负责无数字解释。
7. 打开 Knowledge 页面，检索“最佳 ROC-AUC 模型”，展示路径、章节、分数和内容哈希。
8. 打开 Evidence 页面，选择一个 Claim，查看 runs、commit、配置哈希和 artifacts。
9. 演示未审批时同步被阻断；由 Ethan 审批后同步两个 `paper/` 证据文件。
10. 切换到故障 Trace，说明 resume 只重试失败任务；以 Day 7 eval 和发布材料收尾。
