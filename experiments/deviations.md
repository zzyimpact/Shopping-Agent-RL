# Current single_persona dataset deviation

The historical P3b-v1.1 collection plan targeted 6000 successful trajectories
per scenario (approximately two demonstrations per task) across two active
scenarios. The recorded run `formal-20260909T035508Z-17af7151` completed Pass A
for single_persona, reached 3000/3000 coverage slots, and was stopped before
Pass B/C when the teacher API budget was exhausted. Its collection status and
raw accepted artifacts remain historical and unchanged.

The current offline decision is a budgeted dataset freeze for
`single_persona` only. It starts from 2673 candidate accepted unique tasks and
freezes 2668 after offline terminal-integrity and target-id-exploitation gates;
there is one demonstration per task. Teacher models are retained as recorded.

`persona-policy-sanitizer-v1` removes identifier-only persona fields (including
`用户ID`) from policy-visible prompts. This changes the Persona policy-observation
contract; the historical Persona fixed-128 result is therefore historical only.
Future Base, SFT, Direct GRPO, and SFT→GRPO Persona runs must use this same
sanitizer version and a newly collected/evaluated sanitized Persona baseline.
