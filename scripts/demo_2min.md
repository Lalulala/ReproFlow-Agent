# ReproFlow Agent 两分钟演示讲稿

| 时间 | 画面与讲解 |
| --- | --- |
| 0:00–0:15 | 打开“对话实验”，输入：找到主实验入口，复现 baseline，并用三个固定种子比较新方法。 |
| 0:15–0:35 | 展示 Agent 的仓库阅读范围、Dependency Preflight、代码 Diff、运行矩阵和指标规则；强调此时零写入、零执行。 |
| 0:35–0:50 | 由 Ethan 审批计划；说明执行使用 `shell=False`、命令白名单、路径限制和项目隔离 uv 环境。 |
| 0:50–1:05 | 执行实验，展示独立运行目录、日志、metrics、manifest 和 LangGraph checkpoint。 |
| 1:05–1:20 | 打开“实验结果”，按实验选择，展示 summary、aggregate、mean/std、失败信息和中文报告。 |
| 1:20–1:35 | 打开“证据库”，展示 Claim 对应的 run、metric、commit、配置哈希和 artifact；未审批同步会被阻断。 |
| 1:35–1:47 | 打开“知识与记忆”，按实验检索 report、lesson 和失败模式，展示来源与内容哈希。 |
| 1:47–1:55 | 切换到失败实验，展示 Repair Agent 生成的新 Diff 和第二道审批，而不是静默改代码。 |
| 1:55–2:00 | 展示 CI、85% 覆盖率、20/20 Agent eval 和三仓库兼容性矩阵。 |

演示使用 Mock 模式即可完整跑通，不需要 API Key；Repair Agent 片段可使用已保存且已脱敏的
`repo_plans/repo-plan-140393eb5f.md`，不要在录制时批准或执行该历史草稿。
