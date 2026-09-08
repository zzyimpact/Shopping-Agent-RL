# P3a Teacher Profile

本报告由本地 profiling SQLite/JSON artifacts 自动生成；不调用 teacher API，不把旧 contaminated records 当作 canonical 数据。

> P3a 的 3/2 次数是 profiling operational limits，不是 P3b formal collection cap；diversity threshold、reserve policy、并发数均未冻结。

## 1. Data provenance

- Selection version: `p3a-canonical-v1`
- Protocol: `{"environment_version": "task-scoped-v2", "policy_observation_version": "single-eval-policy-v1", "profiler_protocol_version": "p3a-visible-action-v2"}`
- Single: clean 24-task completed run.
- Single&Pers: base 24-task run excluding `934909004241`, `895516447505`, overlaid with the completed 2-task repair run.
- Non-canonical repair runs retained for audit: `[{"run_id": "p3a-repair-20260908T030644Z-49b9a1a8", "status": "stopped", "reason": "non-canonical repair run; retained for audit"}]`

## Single

- Tasks: 24; genuine attempts: 72; infrastructure interruptions excluded: 2; provider/config errors excluded: 1
- First-attempt success (Wilson 95% CI): 0.792 [0.595, 0.908]
- Success within 2 / 3 attempts: 0.792 [0.595, 0.908] / 0.792 [0.595, 0.908]
- Eventual first-success: 0.792 [0.595, 0.908]; profile_unsolved: 5
- First-success attempts: 34; attempts/success: 1; distribution: `{"1": 19}`
- Second-demo: 19 tasks entered, 38 attempts, 38 successes; at-least-one-success: 1.000 [0.832, 1.000]; success/attempt: 1.000 [0.908, 1.000]
- Exact duplicate pairs: 7; first-vs-second duplicate pairs: 7; provisional near-duplicate pairs: 14 (diagnostic only)
- Action length mean/p50/p90/max: 5.298/4.000/7.000/23.000
- Failure modes: `{"max_steps": 5, "profile_unsolved": 5, "success": 57, "terminal_unsuccessful": 5}`

- successful: wall p50/p90/max=38.838s/70.626s/136.977s; API p50/p90/max=7.471s/11.589s/57.454s; env-step p50/p90/max=0.136s/0.283s/2.570s; API wall proportion=0.979; tokens mean input/output=35883/854
- genuine_failed: wall p50/p90/max=223.935s/319.523s/399.432s; API p50/p90/max=7.372s/13.018s/96.876s; env-step p50/p90/max=0.140s/0.272s/1.556s; API wall proportion=0.979; tokens mean input/output=342316/3839
- infrastructure_interrupted: wall p50/p90/max=190.977s/476.231s/476.231s; API p50/p90/max=7.994s/11.770s/18.472s; env-step p50/p90/max=0.237s/0.357s/0.627s; API wall proportion=0.269; tokens mean input/output=97397/1536
- Behavioral feature equal-pair counts (action/search/clicked-product/options/final-purchase): `{"clicked_products": 55, "final_purchase_asin": 57, "normalized_actions": 7, "search_queries": 7, "selected_options": 40}`
- Infrastructure: retries=1; HTTP statuses={"200": 25}
- Hard tasks: `[{"task_id": "681783199335", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 1, "max_steps": 1}, "max_steps_attempts": 1, "repeated_search": true, "repeated_click": true, "action_count": 61, "target_product_clicked": true, "last_reward": {}}, {"task_id": "921111560004", "outcomes": {"profile_unsolved": 1, "max_steps": 2}, "max_steps_attempts": 2, "repeated_search": true, "repeated_click": true, "action_count": 90, "target_product_clicked": true, "last_reward": {}}, {"task_id": "904644781114", "outcomes": {"profile_unsolved": 1, "max_steps": 2}, "max_steps_attempts": 2, "repeated_search": true, "repeated_click": true, "action_count": 90, "target_product_clicked": false, "last_reward": {}}, {"task_id": "761341816374", "outcomes": {"terminal_unsuccessful": 2, "profile_unsolved": 1}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 19, "target_product_clicked": true, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.8333333333333334, "r_option": 0.5, "r_price": 1.0, "r_strict": 0.5, "r_succ": 0.0}}, {"task_id": "632290461063", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 66, "target_product_clicked": false, "last_reward": {"r_attribute": 0.75, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.6666666666666666, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}}]`

## Single & Personalization

- Tasks: 24; genuine attempts: 68; infrastructure interruptions excluded: 0; provider/config errors excluded: 0
- First-attempt success (Wilson 95% CI): 0.750 [0.551, 0.880]
- Success within 2 / 3 attempts: 0.750 [0.551, 0.880] / 0.792 [0.595, 0.908]
- Eventual first-success: 0.792 [0.595, 0.908]; profile_unsolved: 5
- First-success attempts: 30; attempts/success: 1.105; distribution: `{"1": 18, "3": 1}`
- Second-demo: 19 tasks entered, 38 attempts, 34 successes; at-least-one-success: 0.947 [0.754, 0.991]; success/attempt: 0.895 [0.759, 0.958]
- Exact duplicate pairs: 6; first-vs-second duplicate pairs: 6; provisional near-duplicate pairs: 14 (diagnostic only)
- Action length mean/p50/p90/max: 5.491/4.000/7.000/24.000
- Failure modes: `{"profile_unsolved": 3, "success": 53, "terminal_unsuccessful": 12}`

- successful: wall p50/p90/max=31.840s/88.333s/190.256s; API p50/p90/max=5.361s/9.900s/120.579s; env-step p50/p90/max=0.132s/0.209s/1.892s; API wall proportion=0.980; tokens mean input/output=37871/598
- genuine_failed: wall p50/p90/max=40.484s/76.943s/83.828s; API p50/p90/max=5.439s/8.891s/65.321s; env-step p50/p90/max=0.133s/0.220s/1.783s; API wall proportion=0.974; tokens mean input/output=39457/675
- infrastructure_interrupted: wall p50/p90/max=N/A/N/A/N/A; API p50/p90/max=N/A/N/A/N/A; env-step p50/p90/max=N/A/N/A/N/A; API wall proportion=N/A; tokens mean input/output=N/A/N/A
- Behavioral feature equal-pair counts (action/search/clicked-product/options/final-purchase): `{"clicked_products": 44, "final_purchase_asin": 48, "normalized_actions": 6, "search_queries": 7, "selected_options": 38}`
- Infrastructure: retries=1; HTTP statuses={}
- Hard tasks: `[{"task_id": "706570359072", "outcomes": {"terminal_unsuccessful": 2, "profile_unsolved": 1}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 24, "target_product_clicked": false, "last_reward": {"r_attribute": 0.6, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.5714285714285714, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}}, {"task_id": "904385584071", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 17, "target_product_clicked": false, "last_reward": {"r_attribute": 0.14285714285714285, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.2222222222222222, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}}, {"task_id": "933047766034", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 14, "target_product_clicked": true, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.875, "r_option": 0.5, "r_price": 1.0, "r_strict": 0.5, "r_succ": 0.0}}]`

## Formal collection implications

这些是 P3b 的数据输入，不是已冻结决策：API latency dominates serial wall time；hard-task retries can be disproportionately costly；24-task estimates have wide uncertainty；Persona project pool is 3323 rather than paper 3383。Formal caps, diversity threshold, reserve replacement and concurrency remain open for P3b。

串行 rough projection（假设行为和延迟稳定）：

- Single: 3000-task initial unique-success estimate=2375.000 (Wilson scaled range 1785.886–2722.655); 6000 successes≈81.270h; 12000 total≈162.539h。
- Single&Pers: 3000-task initial unique-success estimate=2375.000 (Wilson scaled range 1785.886–2722.655); 6000 successes≈76.014h; 12000 total≈152.028h。

## Excluded history

旧 implementation/config/resume 污染记录保留在本地 audit artifacts 中，不进入 canonical summary；不应作为 teacher failure 或 success rate denominator。
