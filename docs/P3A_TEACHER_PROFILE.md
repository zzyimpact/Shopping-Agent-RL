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
- Solved-task attempts-to-success mean: 1; distribution: `{"1": 19}`
- First-phase attempt burden / acquired success: 1.789
- Second-demo: 19 tasks entered, 38 attempts, 38 successes; at-least-one-success: 1.000 [0.832, 1.000]; success/attempt: 1.000 [0.908, 1.000]
- Exact duplicates: all 7/57 (0.123); first-second 6/38 (0.158); second-second 1/19 (0.053)
- Provisional near-duplicates: 14/57 (0.246); diagnostic only, not a formal threshold
- Action length mean/p50/p90/max: 5.298/4.000/7.000/23.000
- Failure modes: `{"attempt_outcomes": {"max_steps": 5, "profile_unsolved": 5, "success": 57, "terminal_unsuccessful": 5}, "hard_tasks": [{"action_count": 61, "last_reward": {}, "max_steps_attempts": 1, "outcomes": {"max_steps": 1, "profile_unsolved": 1, "terminal_unsuccessful": 1}, "repeated_click": true, "repeated_search": true, "target_product_clicked": true, "task_id": "681783199335"}, {"action_count": 90, "last_reward": {}, "max_steps_attempts": 2, "outcomes": {"max_steps": 2, "profile_unsolved": 1}, "repeated_click": true, "repeated_search": true, "target_product_clicked": true, "task_id": "921111560004"}, {"action_count": 90, "last_reward": {}, "max_steps_attempts": 2, "outcomes": {"max_steps": 2, "profile_unsolved": 1}, "repeated_click": true, "repeated_search": true, "target_product_clicked": false, "task_id": "904644781114"}, {"action_count": 19, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.8333333333333334, "r_option": 0.5, "r_price": 1.0, "r_strict": 0.5, "r_succ": 0.0}, "max_steps_attempts": 0, "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "repeated_click": true, "repeated_search": true, "target_product_clicked": true, "task_id": "761341816374"}, {"action_count": 66, "last_reward": {"r_attribute": 0.75, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.6666666666666666, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}, "max_steps_attempts": 0, "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "repeated_click": true, "repeated_search": true, "target_product_clicked": false, "task_id": "632290461063"}], "task_outcomes": {"profile_unsolved": 5, "solved": 19}, "termination": {"max_steps": 5, "profile_unsolved": 5, "success": 57, "terminal_unsuccessful": 5}}`

- successful: trajectory-attempt-wall p50/p90/max=38.838s/70.626s/136.977s; API-call latency p50/p90/max=7.471s/11.589s/57.454s; API-total/attempt p50/p90/max=38.064s/69.322s/131.848s; env-step p50/p90/max=0.136s/0.283s/2.570s; API wall proportion=0.979; tokens mean input/output=35883/854
- genuine_failed: trajectory-attempt-wall p50/p90/max=223.935s/319.523s/399.432s; API-call latency p50/p90/max=7.372s/13.018s/96.876s; API-total/attempt p50/p90/max=220.878s/313.543s/393.994s; env-step p50/p90/max=0.140s/0.272s/1.556s; API wall proportion=0.979; tokens mean input/output=342316/3839
- infrastructure_interrupted: trajectory-attempt-wall p50/p90/max=190.977s/476.231s/476.231s; API-call latency p50/p90/max=7.994s/11.770s/18.472s; API-total/attempt p50/p90/max=33.464s/179.797s/179.797s; env-step p50/p90/max=0.237s/0.357s/0.627s; API wall proportion=0.269; tokens mean input/output=97397/1536
- Behavioral feature equal-pair counts (action/search/clicked-product/options/final-purchase): `{"clicked_products": 55, "final_purchase_asin": 57, "normalized_actions": 7, "search_queries": 7, "selected_options": 40}`
- Infrastructure: retries=4; retry events=0; telemetry complete=False; HTTP statuses={}
- Hard tasks: `[{"task_id": "681783199335", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 1, "max_steps": 1}, "max_steps_attempts": 1, "repeated_search": true, "repeated_click": true, "action_count": 61, "target_product_clicked": true, "last_reward": {}}, {"task_id": "921111560004", "outcomes": {"profile_unsolved": 1, "max_steps": 2}, "max_steps_attempts": 2, "repeated_search": true, "repeated_click": true, "action_count": 90, "target_product_clicked": true, "last_reward": {}}, {"task_id": "904644781114", "outcomes": {"profile_unsolved": 1, "max_steps": 2}, "max_steps_attempts": 2, "repeated_search": true, "repeated_click": true, "action_count": 90, "target_product_clicked": false, "last_reward": {}}, {"task_id": "761341816374", "outcomes": {"terminal_unsuccessful": 2, "profile_unsolved": 1}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 19, "target_product_clicked": true, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.8333333333333334, "r_option": 0.5, "r_price": 1.0, "r_strict": 0.5, "r_succ": 0.0}}, {"task_id": "632290461063", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 66, "target_product_clicked": false, "last_reward": {"r_attribute": 0.75, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.6666666666666666, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}}]`

## Single & Personalization

- Tasks: 24; genuine attempts: 75; infrastructure interruptions excluded: 0; provider/config errors excluded: 0
- First-attempt success (Wilson 95% CI): 0.750 [0.551, 0.880]
- Success within 2 / 3 attempts: 0.792 [0.595, 0.908] / 0.833 [0.641, 0.933]
- Eventual first-success: 0.833 [0.641, 0.933]; profile_unsolved: 4
- Solved-task attempts-to-success mean: 1.150; distribution: `{"1": 18, "2": 1, "3": 1}`
- First-phase attempt burden / acquired success: 1.750
- Second-demo: 20 tasks entered, 40 attempts, 36 successes; at-least-one-success: 0.950 [0.764, 0.991]; success/attempt: 0.900 [0.769, 0.960]
- Exact duplicates: all 6/53 (0.113); first-second 4/36 (0.111); second-second 2/17 (0.118)
- Provisional near-duplicates: 14/53 (0.264); diagnostic only, not a formal threshold
- Action length mean/p50/p90/max: 6.482/4.000/10.000/26.000
- Failure modes: `{"attempt_outcomes": {"malformed_action": 1, "profile_unsolved": 4, "success": 56, "terminal_unsuccessful": 14}, "hard_tasks": [{"action_count": 24, "last_reward": {"r_attribute": 0.6, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.5714285714285714, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}, "max_steps_attempts": 0, "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "repeated_click": true, "repeated_search": true, "target_product_clicked": false, "task_id": "706570359072"}, {"action_count": 17, "last_reward": {"r_attribute": 0.14285714285714285, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.2222222222222222, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}, "max_steps_attempts": 0, "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "repeated_click": true, "repeated_search": true, "target_product_clicked": false, "task_id": "904385584071"}, {"action_count": 14, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.875, "r_option": 0.5, "r_price": 1.0, "r_strict": 0.5, "r_succ": 0.0}, "max_steps_attempts": 0, "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "repeated_click": true, "repeated_search": true, "target_product_clicked": true, "task_id": "933047766034"}, {"action_count": 36, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.7777777777777778, "r_option": 0.3333333333333333, "r_price": 1.0, "r_strict": 0.3333333333333333, "r_succ": 0.0}, "max_steps_attempts": 0, "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "repeated_click": true, "repeated_search": true, "target_product_clicked": true, "task_id": "934909004241"}], "task_outcomes": {"profile_unsolved": 4, "solved": 20}, "termination": {"malformed_action": 1, "profile_unsolved": 4, "success": 56, "terminal_unsuccessful": 14}}`

- successful: trajectory-attempt-wall p50/p90/max=32.841s/103.445s/228.871s; API-call latency p50/p90/max=5.374s/9.879s/120.579s; API-total/attempt p50/p90/max=32.069s/100.494s/223.505s; env-step p50/p90/max=0.133s/0.229s/1.892s; API wall proportion=0.979; tokens mean input/output=55872/701
- genuine_failed: trajectory-attempt-wall p50/p90/max=53.925s/83.828s/105.851s; API-call latency p50/p90/max=5.473s/10.427s/65.321s; API-total/attempt p50/p90/max=51.552s/83.131s/101.278s; env-step p50/p90/max=0.130s/0.225s/2.334s; API wall proportion=0.974; tokens mean input/output=48493/830
- infrastructure_interrupted: trajectory-attempt-wall p50/p90/max=N/A/N/A/N/A; API-call latency p50/p90/max=N/A/N/A/N/A; API-total/attempt p50/p90/max=N/A/N/A/N/A; env-step p50/p90/max=N/A/N/A/N/A; API wall proportion=N/A; tokens mean input/output=N/A/N/A
- Behavioral feature equal-pair counts (action/search/clicked-product/options/final-purchase): `{"clicked_products": 44, "final_purchase_asin": 51, "normalized_actions": 6, "search_queries": 7, "selected_options": 38}`
- Infrastructure: retries=2; retry events=0; telemetry complete=False; HTTP statuses={}
- Hard tasks: `[{"task_id": "706570359072", "outcomes": {"terminal_unsuccessful": 2, "profile_unsolved": 1}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 24, "target_product_clicked": false, "last_reward": {"r_attribute": 0.6, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.5714285714285714, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}}, {"task_id": "904385584071", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 17, "target_product_clicked": false, "last_reward": {"r_attribute": 0.14285714285714285, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.2222222222222222, "r_option": 0.0, "r_price": 1.0, "r_strict": 0.0, "r_succ": 0.0}}, {"task_id": "933047766034", "outcomes": {"profile_unsolved": 1, "terminal_unsuccessful": 2}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 14, "target_product_clicked": true, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.875, "r_option": 0.5, "r_price": 1.0, "r_strict": 0.5, "r_succ": 0.0}}, {"task_id": "934909004241", "outcomes": {"terminal_unsuccessful": 2, "profile_unsolved": 1}, "max_steps_attempts": 0, "repeated_search": true, "repeated_click": true, "action_count": 36, "target_product_clicked": true, "last_reward": {"r_attribute": 1.0, "r_category": 1.0, "r_finish": 1.0, "r_loose": 0.7777777777777778, "r_option": 0.3333333333333333, "r_price": 1.0, "r_strict": 0.3333333333333333, "r_succ": 0.0}}]`

## Formal collection implications

这些是 P3b 的数据输入，不是已冻结决策：API latency dominates serial wall time；hard-task retries can be disproportionately costly；24-task estimates have wide uncertainty；Persona project pool is 3323 rather than paper 3383。Formal caps, diversity threshold, reserve replacement and concurrency remain open for P3b。

串行 rough projection（假设行为和延迟稳定）：

- Single: 3000-task initial unique-success estimate=2375.000 (Wilson scaled range 1785.886–2722.655); success-only lower-bound 6000≈81.270h; observed P3a cost-aware 6000≈169.156h。
- Single&Pers: 3000-task initial unique-success estimate=2500.000 (Wilson scaled range 1924.408–2799.640); success-only lower-bound 6000≈87.714h; observed P3a cost-aware 6000≈117.708h。

## Excluded history

旧 implementation/config/resume 污染记录保留在本地 audit artifacts 中，不进入 canonical summary；不应作为 teacher failure 或 success rate denominator。
