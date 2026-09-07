# P3a Teacher Profile

这是 `scripts/summarize_teacher_profile.py` 的输出模板说明。Codex 未调用真实 teacher API，
因此本仓库不包含 profiling 数字；用户运行两个 scenario 后，使用 summarizer 生成本地报告。

报告应覆盖每个 scenario 的 first-attempt/eventual success、attempts/first-success、second
success、exact/provisional near duplicate、action length、termination、malformed/invalid/
max-step、API latency/retry/token usage、environment latency 和 infrastructure interruption。

P3a 的 3/2 attempt limits 是 profiling-only operational limits，不是 P3b formal collection
policy；diversity 只使用行为 fingerprint，不把 Thought 文案差异当作策略差异。
