from __future__ import annotations

from collections.abc import Iterable

from .models import EvidenceClaim, ExperimentPlan

PLAN_STATUS_LABELS = {
    "draft": "待审核",
    "approved": "已批准，可执行",
    "rejected": "已驳回",
    "completed": "已完成",
}

CLAIM_STATUS_LABELS = {
    "proposed": "待审核",
    "supported": "已审核：实验支持",
    "contradicted": "已审核：实验反驳",
    "inconclusive": "已审核：证据不足",
    "stale": "已失效：需重新实验",
}

PROPOSED_STATUS_LABELS = {
    "supported": "实验支持该主张",
    "contradicted": "实验反驳该主张",
    "inconclusive": "当前证据不足",
}

VARIANT_LABELS = {
    "logistic_regression": "逻辑回归（Logistic Regression）",
    "random_forest": "随机森林（Random Forest）",
    "svm": "支持向量机（SVM）",
}

METRIC_LABELS = {
    "accuracy": "准确率（Accuracy）",
    "f1": "F1 分数",
    "roc_auc": "ROC-AUC",
}


def _variant(name: str) -> str:
    return VARIANT_LABELS.get(name, name)


def _metric(name: str) -> str:
    return METRIC_LABELS.get(name, name)


def _display_hypothesis(plan: ExperimentPlan) -> str:
    bundled = (
        "At least one non-linear model will improve mean ROC-AUC over logistic regression "
        "across three fixed train/test splits."
    )
    if plan.hypothesis == bundled:
        return "在三组固定的训练/测试划分上，至少有一种非线性模型的平均 ROC-AUC 会高于逻辑回归。"
    return plan.hypothesis


def render_plan_markdown(plan: ExperimentPlan) -> str:
    variants = "、".join(_variant(item.name) for item in plan.variants)
    metrics = "、".join(_metric(item.name) for item in plan.metrics)
    seeds = "、".join(str(seed) for seed in plan.seeds)
    run_count = len(plan.variants) * len(plan.seeds)
    if plan.script_path == "examples/sklearn_demo/experiment.py":
        dataset = "sklearn 内置乳腺癌分类数据集"
    elif plan.data_path:
        dataset = plan.data_path
    else:
        dataset = "由实验脚本负责加载"
    approval = (
        f"已由 {plan.approved_by} 审核批准" if plan.approved_by else "尚未审核，审批前不会执行"
    )
    source_note = (
        f"规划时参考了 {len(plan.context_sources)} 条本地知识或历史记忆。"
        if plan.context_sources
        else "本次规划未检索到额外的本地知识或历史记忆。"
    )
    command = " ".join(plan.command)
    return f"""# 实验计划：{plan.goal}

> 当前状态：{PLAN_STATUS_LABELS.get(plan.status.value, plan.status.value)}
> 计划编号：`{plan.plan_id}`

## 一、为什么做这个实验

**实验目标**

{plan.goal}

**待验证假设**

{_display_hypothesis(plan)}

## 二、实验怎么做

本实验使用 **{dataset}**，比较 {variants}。
{_variant(plan.baseline)} 作为基线模型。

每个模型分别使用随机种子 {seeds} 运行，总计 **{run_count} 组实验**。
单次实验最长允许运行 {plan.timeout_seconds} 秒。

## 三、怎么判断结果

系统将记录 {metrics}，并计算每个模型在多个随机种子下的平均值、
标准差、最佳值及相对基线的变化。

## 四、会产生什么

- 每次运行的日志、指标和环境快照。
- 逐次结果 `summary.csv` 和汇总结果 `aggregate.csv`。
- 指标图表和中文 Markdown 实验报告。
- 可供人工审核的论文 Evidence Claim。

## 五、审核与安全

{approval}。{source_note}
系统只会执行白名单内的参数化命令，不使用 Shell，不会自动修改训练代码。

## 六、技术复现信息

- 实验脚本：`{plan.script_path}`
- 基础命令：`{command}`
- 产物根目录：`{plan.artifact_root}/`
- 机器可读计划：`.reproflow/plans/{plan.plan_id}.yaml`
"""


def _claim_conclusion(claim: EvidenceClaim) -> str:
    baseline = _variant(claim.baseline_variant)
    observed = _variant(claim.observed_variant)
    metric = _metric(claim.metric)
    if claim.proposed_status == "supported":
        return f"{observed} 的平均 {metric} 高于 {baseline}，实验支持该比较结论。"
    if claim.proposed_status == "contradicted":
        return f"{observed} 的平均 {metric} 低于 {baseline}，实验反驳了“它更好”的主张。"
    return f"当前实验不足以判断 {observed} 的平均 {metric} 是否优于 {baseline}。"


def render_evidence_markdown(claim: EvidenceClaim, *, include_artifacts: bool = True) -> str:
    delta = f"{claim.delta:+.6f}"
    reviewer = (
        f"{claim.reviewed_by}（{claim.reviewed_at.isoformat()}）"
        if claim.reviewed_by and claim.reviewed_at
        else "尚未人工审核"
    )
    scope = (
        "这表示当前实验支持该结论，但不等于已经证明其具有统计显著性或可以推广到其他数据集。"
        if claim.proposed_status == "supported"
        else "反驳记录也是有效的科研证据，它用来防止后续报告将未被实验支持的说法写成结论。"
    )
    artifacts = ""
    if include_artifacts:
        artifact_lines = "\n".join(f"- `{path}`" for path in claim.artifacts)
        artifacts = f"""

## 溯源文件

{artifact_lines}
"""
    return f"""# 证据记录 {claim.claim_id}

> 状态：{CLAIM_STATUS_LABELS.get(claim.status.value, claim.status.value)}
> 系统建议：{PROPOSED_STATUS_LABELS[claim.proposed_status]}

## 一句话结论

{_claim_conclusion(claim)}

## 这个结论是怎么得到的

实验以 **{_variant(claim.baseline_variant)}** 为基线，其平均 {_metric(claim.metric)} 为
**{claim.baseline_value:.6f}**。对比的 **{_variant(claim.observed_variant)}** 平均值为
**{claim.observed_value:.6f}**，相对基线的变化为 **{delta}**。

该比较使用随机种子 {"、".join(str(seed) for seed in claim.seeds)}，关联
{len(claim.experiment_ids)} 次已记录的实验运行。

## 应该如何理解

{scope}

## 审核与复现信息

- 审核人：{reviewer}
- 工作流：`{claim.workflow_id}`
- Git commit：`{claim.code_commit}`
- 实验配置哈希：`{claim.config_hash}`
- 论文目标章节：{"、".join(claim.paper_sections)}
{artifacts}"""


def render_evidence_registry_markdown(claims: Iterable[EvidenceClaim]) -> str:
    claims = list(claims)
    sections = [
        "# 已审核的实验证据摘要",
        "",
        "> 本文档由 ReproFlow 从已审核的 Evidence Registry 自动生成，请勿手工修改。",
        "",
        f"当前共有 **{len(claims)} 条**已审核证据。"
        "每条证据都可追溯到实验指标、代码版本和运行产物。",
    ]
    for claim in claims:
        section = render_evidence_markdown(claim, include_artifacts=False)
        sections.extend(["", "---", "", section])
    return "\n".join(sections).rstrip() + "\n"
