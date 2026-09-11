"""No training dependencies, model, network, or remote ShopEnv used here."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from env.teacher_env_client import TeacherEnvError
from rewards.shopsim_reward import METRIC_KEYS
from training.grpo import (
    GRPOSpec, build_grpo_trainer, collect_online_batch, group_metadata,
    initialize_grpo_model, load_grpo_config, metadata_bridge, rollout_reward,
    shop_trainer_class, task_dataset_rows, task_schedule, train_grpo, validate_lineage,
)
from training.policy import GenerationConfig, PolicySample, QwenPolicy
from training.rollout import AgentRollout
from tests.training.test_rollout import FakeEnv, step_payload
from tests.training.test_sft_data import FakeQwenTokenizer


SAMPLE = GenerationConfig(do_sample=True, max_new_tokens=8, max_context_tokens=100)
SEARCH = "Thought: search\nAction: search[shoes]"
BUY = "Thought: buy\nAction: click[Buy]"
ROOT = Path(__file__).resolve().parents[2]


class SamplingPolicy:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.prompts, self.inputs, self.observations = [], [], []

    def prompt_token_ids(self, messages, sampling):
        self.prompts.append(deepcopy(messages))
        return [10, 11]

    def observation_token_ids(self, observation, *, previous_token, sampling):
        self.observations.append((observation, previous_token))
        return [70, 71, 72]  # External role/header/observation/prefix tokens.

    def sample(self, input_ids, *, sampling):
        assert sampling.do_sample
        self.inputs.append(list(input_ids))
        return next(self.samples)


def sample(text=BUY, ids=(90, 99), probs=(-0.2, -0.4)):
    return PolicySample(text, list(ids), list(probs))


def test_multiturn_source_ids_masks_logprobs_and_terminal_omission():
    # Text is deliberately unrelated to IDs: re-tokenizing it would fail these assertions.
    policy = SamplingPolicy([sample(SEARCH, (81, 99), (-0.1, -0.3)), sample()])
    env = FakeEnv([step_payload("visible O1"), step_payload("terminal O2", done=True, success=True)])
    episode = AgentRollout(policy=policy, env_factory=lambda: env, scenario="single").run("hidden-task-id", sampling=SAMPLE)
    trace = episode.token_trace
    assert trace.prompt_ids == [10, 11]
    assert trace.completion_ids == [81, 99, 70, 71, 72, 90, 99]
    assert trace.logprobs == [-0.1, -0.3, 0.0, 0.0, 0.0, -0.2, -0.4]
    assert trace.env_mask == [1, 1, 0, 0, 0, 1, 1]  # Sampled EOS remains policy-owned.
    assert len(trace.completion_ids) == len(trace.logprobs) == len(trace.env_mask)
    assert policy.inputs == [[10, 11], [10, 11, 81, 99, 70, 71, 72]]
    assert policy.observations == [("visible O1", 99)]
    assert episode.messages[-1]["content"] == "terminal O2"  # Retained for diagnostics only.
    assert episode.status == "success" and episode.reward == 1
    assert env.closed and env.released
    assert "hidden-task-id" not in json.dumps(policy.prompts)
    assert "PRIVATE" not in json.dumps(policy.prompts)


@pytest.mark.parametrize("failure", ["max_steps", "malformed_action", "invalid_action", "context_limit"])
def test_policy_failures_keep_generated_tokens_without_trailing_observation(failure):
    text = "bad answer" if failure == "malformed_action" else SEARCH
    policy = SamplingPolicy([sample(text)])
    payload = step_payload("unused trailing page", valid=failure != "invalid_action")
    env = FakeEnv([payload])
    settings = GenerationConfig(do_sample=True, max_new_tokens=2, max_context_tokens=6) if failure == "context_limit" else SAMPLE
    episode = AgentRollout(policy=policy, env_factory=lambda: env, scenario="single",
                            max_action_steps=2 if failure == "context_limit" else 1).run("t", sampling=settings)
    assert episode.status == failure and episode.reward == 0
    assert episode.token_trace.completion_ids == [90, 99]
    assert episode.token_trace.env_mask == [1, 1]
    assert len(policy.inputs) == 1
    assert env.closed


def test_infrastructure_failure_propagates_in_capture_mode():
    env = FakeEnv([TeacherEnvError("connection lost", kind="infrastructure")])
    policy = SamplingPolicy([sample(SEARCH)])
    with pytest.raises(TeacherEnvError):
        AgentRollout(policy=policy, env_factory=lambda: env, scenario="single").run("t", sampling=SAMPLE)
    assert env.closed and env.released


def grouped_rows():
    rows = task_dataset_rows(["A-secret-id", "B-secret-id"], "single_persona")
    return [deepcopy(row) for row in rows for _ in range(3)]


def test_g3_two_tasks_metadata_order_independent_sessions_and_reward_passthrough():
    created = []

    class IsolatedEnv(FakeEnv):
        def reset(self, scenario, task_id):
            assert not hasattr(self, "session")  # Exactly one reset on every fresh env instance.
            self.session = f"session-{len(created)}"
            result = super().reset(scenario, task_id)
            result.payload["session_id"] = self.session
            return result

        def step(self, session_id, response, **kwargs):
            assert session_id == self.session
            return super().step("session-1", response, **kwargs)

        def release(self, session_id):
            assert session_id == self.session
            super().release(session_id)

    def factory():
        env = IsolatedEnv([step_payload("terminal", done=True, success=True)])
        created.append(env)
        return env

    rows = grouped_rows()
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=4))
    policy = SamplingPolicy([sample(ids=(80 + i, 99)) for i in range(6)])
    with metadata_bridge(trainer, rows, scenario="single_persona", group_size=3):
        result = collect_online_batch([row["prompt"] for row in rows], trainer, policy=policy,
            env_factory=factory, scenario="single_persona", sampling=SAMPLE,
            spec=GRPOSpec(num_generations=3), reward_alpha=0.5, max_action_steps=30)
    assert trainer._shop_batch is None
    assert [env.task_id for env in created] == ["A-secret-id"] * 3 + ["B-secret-id"] * 3
    assert len({id(env) for env in created}) == 6
    assert len({env.session for env in created}) == 6
    assert all(env.closed and env.released == [env.session] and len(env.responses) == 1 for env in created)
    assert result["completion_ids"] == [[80 + i, 99] for i in range(6)]  # Not 18 completions!
    assert all(set(metrics) == {*METRIC_KEYS, "r_alpha"} for metrics in result["rollout_metrics"])
    assert len(policy.prompts) == 6 and "blue" in str(policy.prompts)
    assert "secret-id" not in str(policy.prompts) and "PRIVATE" not in str(policy.prompts)
    before = len(created)
    assert rollout_reward(**result) == [1.0] * 6
    assert len(created) == before
    assert trainer._shop_last_metrics["task_ids"] == ["A-secret-id"] * 3 + ["B-secret-id"] * 3
    assert trainer._shop_last_metrics["zero_variance_group_fraction"] == 1.0
    assert trainer._shop_last_metrics["generated_policy_tokens"] == 12
    # Complete groups may reorder; metadata comes from the actual input row, never prompt lookup.
    reordered = rows[3:] + rows[:3]
    assert group_metadata(reordered, scenario="single_persona", group_size=3)[0]["task_id"] == "B-secret-id"
    with pytest.raises(ValueError, match="contiguous"):
        group_metadata([rows[i] for i in [0, 3, 1, 4, 2, 5]], scenario="single_persona", group_size=3)
    with pytest.raises(ValueError, match="complete"):
        group_metadata(rows[:-1], scenario="single_persona", group_size=3)


@pytest.mark.parametrize("alpha,expected", [(1.0, 0.5), (0.0, 0.8), (0.5, 0.65)])
def test_online_scalar_reward_endpoints_and_interpolation(alpha, expected):
    terminal = step_payload("terminal", done=True)
    terminal.update(reward=0.8, reward_detail={"r_type": 1, "r_att": 0.5, "r_option": 1, "r_price": 1})
    episode = AgentRollout(policy=SamplingPolicy([sample()]), env_factory=lambda: FakeEnv([terminal]),
                           scenario="single", reward_alpha=alpha).run("t", sampling=SAMPLE)
    assert episode.reward == pytest.approx(expected)
    assert set(episode.reward_metrics) == {*METRIC_KEYS, "r_alpha"}


def test_minimal_subclass_bridge_cleanup_copy_and_parent_error():
    rows = grouped_rows()

    class Parent:
        num_generations = 3
        shop_scenario = "single_persona"

        def _generate_and_score_completions(self, inputs):
            assert self._shop_batch[0]["task_id"] == inputs[0]["task_id"]
            inputs[0]["rollout_reward"] = 123  # TRL merges reward extras into batch dicts.
            if self.fail:
                raise RuntimeError("parent infrastructure error")
            return "parent result"

    trainer = shop_trainer_class(Parent)()
    trainer.fail = False
    assert trainer._generate_and_score_completions(rows) == "parent result"
    assert trainer._shop_batch is None and "rollout_reward" not in rows[0]
    trainer.fail = True
    with pytest.raises(RuntimeError, match="infrastructure"):
        trainer._generate_and_score_completions(rows)
    assert trainer._shop_batch is None


def test_seeded_task_schedule_resume_has_no_private_sampler_rng():
    schedule = task_schedule(["a", "b", "c"], seed=1, groups=2, updates=4)
    assert schedule == task_schedule(["a", "b", "c"], seed=1, groups=2, updates=4)
    assert len(schedule) == 8 and set(schedule) == {"a", "b", "c"}
    rows = task_dataset_rows(schedule, "single")
    # Source-audited RepeatSampler: group chunk -> repeat whole generation batch
    # `steps_per_generation` times -> each task index repeated G times contiguously.
    g, accumulation = 3, 2
    batches = [[row for row in rows[start:start + 2] for _ in range(g)]
               for start in range(0, len(rows), 2) for _ in range(accumulation)]
    checkpoint_step = 2
    resumed_batch = batches[checkpoint_step * accumulation]
    assert [row["schedule_index"] for row in resumed_batch] == [4, 4, 4, 5, 5, 5]
    assert [row["task_id"] for row in resumed_batch] == [schedule[4]] * g + [schedule[5]] * g


class NativeShapeTokenizer(FakeQwenTokenizer):
    eos_token_id, pad_token_id = 200001, 200002

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False, **kwargs):
        self.rendered_messages = deepcopy(messages)
        text = super().apply_chat_template(messages, tokenize=False)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                text += "<think>\n\n</think>\n\n"
        return self(text)["input_ids"] if tokenize else text


@pytest.mark.parametrize("has_eos", [True, False])
def test_external_suffix_native_template_owns_inserted_closure_headers_and_prefix(has_eos):
    tok = NativeShapeTokenizer()
    policy = QwenPolicy(model=object(), tokenizer=tok)
    settings = GenerationConfig(do_sample=True, chat_template_kwargs={"enable_thinking": False})
    previous = tok.eos_token_id if has_eos else 555
    ids = policy.observation_token_ids("visible observation", previous_token=previous, sampling=settings)
    prefix = "\n" if has_eos else "<|im_end|>\n"
    expected = prefix + "<|im_start|>user\nvisible observation<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert ids == tok(expected)["input_ids"]
    assert tok.rendered_messages == [{"role": "user", "content": "visible observation"}]


@pytest.mark.parametrize("temperature,top_p", [(1.0, 1.0), (0.7, 0.9)])
def test_sampling_uses_backend_ids_and_normalized_transition_scores(monkeypatch, temperature, top_p):
    class Vector(list):
        def tolist(self): return list(self)
        def float(self): return self
        def cpu(self): return self

    class Tensor:
        def __init__(self, rows): self.rows = rows
        def __getitem__(self, item):
            return Vector(self.rows[item[0]][item[1]]) if isinstance(item, tuple) else Vector(self.rows[item])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        long="long", tensor=lambda rows, **kwargs: Tensor(rows), ones_like=lambda ids: "attention",
        inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=SimpleNamespace))

    class Model:
        device = "fake-device"

        def generate(self, input_ids, attention_mask, generation_config, use_model_defaults):
            assert use_model_defaults is False
            assert input_ids.rows == [[10, 11]] and attention_mask == "attention"
            assert generation_config.do_sample and generation_config.top_k == 0
            assert generation_config.temperature == temperature and generation_config.top_p == top_p
            assert generation_config.max_new_tokens == 3
            return SimpleNamespace(sequences=Tensor([[10, 11, 4321, 9876, 200001]]), scores="backend-scores")

        def compute_transition_scores(self, sequences, scores, *, normalize_logits):
            assert scores == "backend-scores" and normalize_logits
            return Tensor([[-0.25, -0.75, -1.25]])

    class Tokenizer(NativeShapeTokenizer):
        def decode(self, ids, **kwargs):
            assert ids == [4321, 9876, 200001]
            return BUY  # Deliberately no relation to the IDs produced by __call__.

    policy = QwenPolicy(model=Model(), tokenizer=Tokenizer())
    actual = policy.sample([10, 11], sampling=GenerationConfig(
        do_sample=True, max_context_tokens=5, max_new_tokens=4, temperature=temperature, top_p=top_p))
    assert actual == PolicySample(BUY, [4321, 9876, 200001], [-0.25, -0.75, -1.25])
    assert policy.last_usage == {"input_tokens": 2, "generated_tokens": 3}
    with pytest.raises(ValueError, match="stochastic"):
        policy.sample([10, 11], sampling=GenerationConfig())


def configured(tmp_path):
    config = load_grpo_config()
    config.update(scenario="single", output_dir=str(tmp_path), model_path="base")
    config["grpo"].update(num_generations=3, task_groups_per_update=2, max_steps=4,
                          per_device_train_batch_size=2, lora_r=8, lora_alpha=16, target_modules=["q_proj"])
    return config


def fake_training_modules(monkeypatch, capture):
    class Parent:
        def __init__(self, **kwargs):
            capture.update(kwargs)
            self.args = kwargs["args"]
            self.num_generations = self.args.num_generations
            self.model_wrapped = kwargs["model"]
            self.state = SimpleNamespace(global_step=0)
            self.accelerator = SimpleNamespace(num_processes=1, unwrap_model=lambda model: model)

        def train(self, **kwargs):
            capture["resume"] = kwargs
            return "hf-native-result"

        def save_model(self, path): capture["final"] = path

    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.12.0")
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(GRPOTrainer=Parent, GRPOConfig=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))


def test_trl_construction_explicit_semantics_current_policy_and_native_resume(tmp_path, monkeypatch):
    capture, seen = {}, []
    fake_training_modules(monkeypatch, capture)

    class Model:
        training = True
        def eval(self): self.training = False
        def train(self, mode): self.training = mode

    def collect(prompts, trainer, *, policy, **kwargs):
        assert not policy.model.training
        seen.append(policy.model)
        trainer._shop_last_metrics = {"event": "fake-rollout"}
        if len(seen) == 3:
            raise TeacherEnvError("connection failed", kind="infrastructure")
        return {"fake": True}

    monkeypatch.setattr("training.grpo.collect_online_batch", collect)
    first, second = Model(), Model()
    config = configured(tmp_path)
    trainer = build_grpo_trainer(model=first, tokenizer=object(), dataset=[], config=config,
                                 env_factory=lambda: pytest.fail("no real env"))
    args = capture["args"]
    assert (args.loss_type, args.num_generations, args.beta, args.scale_rewards, args.learning_rate) == ("grpo", 3, 0, "group", 1e-6)
    assert args.gradient_accumulation_steps == args.steps_per_generation == 3
    assert args.num_iterations == 1 and not args.shuffle_dataset and not args.ignore_data_skip
    assert not args.remove_unused_columns and not args.mask_truncated_completions
    assert not args.use_vllm and not args.use_liger_kernel and args.disable_dropout
    assert args.bf16 and not args.save_only_model and args.eval_strategy == "no"
    assert capture["peft_config"]["r"] == 8 and capture["peft_config"]["lora_dropout"] == 0
    assert capture["rollout_func"]([], trainer) == {"fake": True}
    trainer.model_wrapped = second  # Simulates current Trainer policy replacement/restore.
    capture["rollout_func"]([], trainer)
    assert seen == [first, second] and first.training and second.training
    with pytest.raises(TeacherEnvError):
        capture["rollout_func"]([], trainer)
    assert second.training
    assert train_grpo(trainer, resume_from_checkpoint="checkpoint-2") == "hf-native-result"
    assert capture["resume"] == {"resume_from_checkpoint": "checkpoint-2"}
    assert capture["final"] == str(tmp_path / "checkpoints/final")
    config["init"] = "sft_adapter"
    build_grpo_trainer(model=second, tokenizer=object(), dataset=[], config=config, env_factory=None)
    assert capture["peft_config"] is None  # Continue existing trainable adapter, do not stack a new one.


def test_sft_lineage_scenario_isolation_and_trainable_load(tmp_path, monkeypatch):
    root = tmp_path / "sft-run"
    adapter = root / "checkpoints/checkpoint-100"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    identity = {"config": {"mode": "sft", "scenario": "single"},
                "inputs": {"model_path": str(tmp_path / "base"), "tokenizer_path": str(tmp_path / "base")}}
    manifest = root / "run_manifest.json"
    manifest.write_text(json.dumps({"identity": identity}))
    config = configured(tmp_path)
    config.update(init="sft_adapter", model_path=str(tmp_path / "base"),
                  sft_run_dir=str(root), adapter_path=str(adapter))
    assert validate_lineage(config)["init"] == "sft_adapter"
    with pytest.raises(ValueError, match="scenario"):
        validate_lineage({**config, "scenario": "single_persona"})
    with pytest.raises(ValueError, match="fresh LoRA"):
        validate_lineage({**config, "init": "base"})
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append((a, kw)) or "base")))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append((a, kw)) or "trainable-adapter")))
    assert initialize_grpo_model("base-path", init="base") == "base"
    assert initialize_grpo_model("base-path", init="sft_adapter", adapter_path="adapter-path") == "trainable-adapter"
    assert calls[-1] == (("base", "adapter-path"), {"is_trainable": True, "local_files_only": True})


def test_cli_reward_exclusivity_defaults_preflight_config_and_resume():
    parse = runpy.run_path(str(ROOT / "scripts/train_grpo.py"))["parse_config"]
    base = ["--scenario", "single", "--model-path", "base", "--train-task-manifest", "train.json", "--output-dir", "run",
            "--per-device-train-batch-size", "2", "--lora-r", "8", "--lora-alpha", "16", "--target-modules", "q_proj"]
    config, spec, resume = parse(base)
    assert spec.num_generations == 8 and spec.loss_type == "grpo" and spec.accumulation_steps == 128
    assert config["sampling"]["do_sample"] and config["reward_alpha"] == 1
    for flags, expected in [(["--reward", "strict"], 1), (["--reward", "loose"], 0), (["--alpha", "0.5"], 0.5)]:
        config, spec, resume = parse(base + flags + ["--num-generations", "2", "--resume-from-checkpoint", "checkpoint-1"])
        assert config["reward_alpha"] == expected and "reward" not in config and "alpha" not in config
        assert spec.num_generations == 2 and resume == "checkpoint-1"
    with pytest.raises(SystemExit):
        parse(base + ["--reward", "strict", "--alpha", "0.5"])
    with pytest.raises(SystemExit):
        parse(base + ["--alpha", "1.5"])
    with pytest.raises(ValueError, match="GPU-PREFLIGHT"):
        GRPOSpec().validate("base")
