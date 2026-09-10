"""Opt-in real packages + tiny CPU Qwen3. No network/environment/model downloads.

RUN_TRAINING_PREFLIGHT=1 ... python -m pytest tests/training -q
All generated checkpoints/reports remain in gitignored .cache/d1-preflight/.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_TRAINING_PREFLIGHT") != "1",
                                reason="requires isolated training-preflight environment")
ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT.parent / "models/Qwen3-8B"
ARTIFACTS = ROOT / ".cache/d1-preflight"


def report(name, data):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def runtime():
    import importlib
    from importlib.metadata import version
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    versions = {name: version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")}
    assert versions["trl"] == "1.12.0"
    for name in versions:
        importlib.import_module(name)
    for name in ("policy", "rollout", "sft_data", "sft", "grpo"):
        importlib.import_module("training." + name)
    import platform
    report("environment", {"python": platform.python_version(), "packages": versions, "device": "cpu"})
    return AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)


def tiny_model(tokenizer):
    from transformers import Qwen3Config, Qwen3ForCausalLM, set_seed
    set_seed(17)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=2048, tie_word_embeddings=True,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = Qwen3ForCausalLM(config).float().cpu()
    assert next(model.parameters()).device.type == "cpu"
    assert sum(p.numel() for p in model.parameters()) < 3_000_000
    return model


def adapter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def test_real_tokenizer_four_review_copies(runtime):
    from training.sft_data import tokenize_with_assistant_mask
    paths = [ROOT / ".cache/round_b_review" / name for name in (
        "single-944750557106.json", "single-941565736582.json",
        "single_persona-922242361311.json", "single_persona-740656872880.json")]
    rows = []
    for path in paths:
        original = path.read_bytes()
        example = json.loads(original)
        messages = example["messages"]
        row = tokenize_with_assistant_mask(runtime, messages)
        text = runtime.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        final = text.rfind("<|im_start|>assistant\n")
        end = text.index("<|im_end|>", final) + len("<|im_end|>")
        encoded = runtime(text[:end], add_special_tokens=False, return_offsets_mapping=True)
        assert row["input_ids"] == encoded["input_ids"]
        assert row["input_ids"][-1] == row["labels"][-1] == runtime.eos_token_id
        assert len(row["input_ids"]) < len(runtime.apply_chat_template(messages, tokenize=True))
        # Independently locate assistant spans; no approximate token-length mask.
        spans, cursor = [], 0
        for message in messages:
            header = f"<|im_start|>{message['role']}\n"
            start = cursor + len(header)
            stop = start + len(message["content"]) + len("<|im_end|>")
            if message["role"] == "assistant":
                spans.append((start, stop))
            cursor = stop + 1
        for token, label, (lo, hi) in zip(row["input_ids"], row["labels"], encoded["offset_mapping"]):
            belongs = hi > lo and any(start <= lo < hi <= stop for start, stop in spans)
            assert label == (token if belongs else -100)
        assert sum(label == runtime.eos_token_id for label in row["labels"]) == len(spans)
        with pytest.raises(ValueError, match="silent truncation"):
            tokenize_with_assistant_mask(runtime, messages, max_length=len(row["input_ids"]) - 1)
        assert path.read_bytes() == original
        trained = sum(label != -100 for label in row["labels"])
        rows.append({"example": path.name, "total_tokens": len(row["input_ids"]),
                     "trainable_tokens": trained, "masked_tokens": len(row["input_ids"]) - trained,
                     "assistant_eos": len(spans), "final_token": runtime.convert_ids_to_tokens(row["input_ids"][-1]),
                     "context_32768": "PASS", "source_copy_sha256": hashlib.sha256(original).hexdigest()})
    report("tokenizer", rows)


def stop_after_one():
    from transformers import TrainerCallback
    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
    return Stop()


def checkpoint_states(checkpoint):
    import torch
    assert (checkpoint / "trainer_state.json").is_file()
    assert (checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    assert optimizer["state"] and scheduler["last_epoch"] == 1
    return optimizer, scheduler


def test_real_sft_update_checkpoint_resume(runtime, tmp_path):
    import torch
    from datasets import Dataset
    from training.sft import SFTConfigSpec, build_sft_trainer, train_sft
    from training.sft_data import tokenize_with_assistant_mask
    from training.runtime import prepare_run

    messages = [{"role": "system", "content": "Shop."}, {"role": "user", "content": "Buy."},
                {"role": "assistant", "content": "Thought: OK\nAction: click[buy now]"},
                {"role": "user", "content": "terminal omitted"}]
    row = tokenize_with_assistant_mask(runtime, messages)
    dataset = Dataset.from_list([row])
    spec = SFTConfigSpec(output_dir=tmp_path, num_train_epochs=2, effective_batch_size=1,
                        per_device_train_batch_size=1, gradient_accumulation_steps=1,
                        lora_r=2, lora_alpha=4, target_modules=("q_proj", "v_proj"),
                        gradient_checkpointing=False, save_steps=1, logging_steps=1, learning_rate=1e-3)
    trainer = build_sft_trainer(model=tiny_model(runtime), tokenizer=runtime, train_dataset=dataset,
                                config=spec, use_cpu=True)
    assert trainer.args.device.type == "cpu"
    batch = trainer.data_collator([row])
    assert batch["labels"].tolist()[0] == row["labels"]
    assert not trainer.args.assistant_only_loss and not trainer.args.completion_only_loss
    trainer.model.eval()
    with torch.no_grad():
        output = trainer.model(**batch)
        # TRL's native chunked CE returns logits=None with labels, so get the
        # unchanged labels-free model logits for an independent masked-CE reference.
        logits = trainer.model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, len(runtime)),
                                                      batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        assert torch.allclose(output.loss, expected)
    trainer.model.train()
    before = adapter_hash(trainer.model)
    trainer.add_callback(stop_after_one())
    train_sft(trainer)
    after = adapter_hash(trainer.model)
    assert before != after and trainer.state.global_step == 1
    checkpoint = tmp_path / "checkpoints/checkpoint-1"
    checkpoint_states(checkpoint)
    resumed = build_sft_trainer(model=tiny_model(runtime), tokenizer=runtime, train_dataset=dataset,
                                config=spec, use_cpu=True)
    loaded = []
    from transformers import TrainerCallback
    class Loaded(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            loaded.append((state.global_step, adapter_hash(resumed.model), resumed.lr_scheduler.last_epoch))
            assert resumed.optimizer.state
    resumed.add_callback(Loaded())
    train_sft(resumed, resume_from_checkpoint=checkpoint)
    assert loaded == [(1, after, 1)] and resumed.state.global_step == 2
    assert all(torch.isfinite(torch.tensor(log["loss"])) for log in resumed.state.log_history if "loss" in log)
    with pytest.raises(ValueError, match="final adapter"):
        prepare_run(tmp_path, config={}, inputs={}, resume_from_checkpoint=tmp_path / "checkpoints/final")
    report("sft", {"construction": "PASS", "masked_ce": "PASS", "update": "PASS",
                   "before_hash": before, "after_hash": after, "resume_loaded_step": loaded[0][0],
                   "resume_final_step": resumed.state.global_step, "optimizer_scheduler_restored": True,
                   "checkpoint": str(checkpoint)})


def test_real_unconstrained_sampling(runtime):
    import torch
    from training.policy import QwenPolicy, GenerationConfig
    model = tiny_model(runtime).eval()
    # Real Qwen3 artifacts have non-global defaults. HF >=4.50 otherwise replaces
    # explicitly requested temperature=1/top_p=1 with these artifact values.
    model.generation_config.temperature = 0.6
    model.generation_config.top_p = 0.95
    model.generation_config.top_k = 20
    model.generation_config.repetition_penalty = 1.1
    effective = []
    prepare = model._prepare_generation_config
    def prepare_config(*args, **kwargs):
        config, rest = prepare(*args, **kwargs)
        effective.append((config.temperature, config.top_p, config.top_k, config.repetition_penalty))
        return config, rest
    model._prepare_generation_config = prepare_config
    policy = QwenPolicy(model=model, tokenizer=runtime)
    sampling = GenerationConfig(do_sample=True, max_new_tokens=3, max_context_tokens=128)
    inputs = policy.prompt_token_ids([{"role": "user", "content": "Hi"}], sampling)
    captured = []
    original = model.generate
    def generate(**kwargs):
        result = original(**kwargs)
        captured.append(result)
        return result
    model.generate = generate
    sample = policy.sample(inputs, sampling=sampling)
    assert effective == [(1.0, 1.0, 0, 1.0)]
    result = captured[0]
    assert sample.token_ids == result.sequences[0, len(inputs):].tolist()
    expected = model.compute_transition_scores(result.sequences, result.scores, normalize_logits=True)[0]
    assert torch.allclose(torch.tensor(sample.logprobs), expected)
    assert len(sample.token_ids) == len(sample.logprobs) == 3 and torch.isfinite(expected).all()
    report("sampling", {"token_ids": sample.token_ids, "logprobs": sample.logprobs,
                        "effective_sampling": effective[0],
                        "backend_ids_exact": True, "device": "cpu"})


def test_real_grpo_groups_masks_update_and_resume(runtime, tmp_path, monkeypatch):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from env.teacher_env_client import TeacherEnvError
    from tests.training.test_rollout import FakeEnv, step_payload
    from training.grpo import build_grpo_trainer, load_grpo_config, task_dataset_rows, train_grpo
    from training.runtime import json_hash, prepare_run

    config = load_grpo_config()
    config.update(scenario="single", output_dir=str(tmp_path))
    config["grpo"].update(num_generations=2, task_groups_per_update=2, max_steps=2,
                          per_device_train_batch_size=1, lora_r=2, lora_alpha=4,
                          target_modules=["q_proj", "v_proj"], save_steps=1, learning_rate=1e-3)
    config["sampling"].update(max_new_tokens=32, max_context_tokens=2048)
    schedule = ["TASK_A_PRIVATE", "TASK_B_PRIVATE", "TASK_A_PRIVATE", "TASK_B_PRIVATE"]
    dataset = Dataset.from_list(task_dataset_rows(schedule, "single"))
    identity = {"task_schedule_sha256": json_hash(schedule)}
    prepare_run(tmp_path, config=config, inputs=identity)
    envs, rewards, batches, loss_checks, next_models = [], [], [], [], []

    class Env(FakeEnv):
        def __init__(self):
            self.member = len(envs) % 2
            terminal = step_payload("TERMINAL_NOT_IN_TOKENS", done=True, success=self.member == 0)
            super().__init__([step_payload("EXTERNAL_OBSERVATION_O1"), terminal])
            envs.append(self)

        def reset(self, scenario, task_id):
            assert not hasattr(self, "task_id")
            return super().reset(scenario, task_id)

    def instrument(trainer):
        original_generate = trainer.model.generate

        def generate(**kwargs):
            # Fixture-only constrained real sampling makes a random tiny model produce
            # parseable actions/EOS. Unconstrained sampling is tested separately above.
            # No fake PolicySample or tokenizer round-trip substitutes backend IDs.
            env = envs[-1]
            conditioning = runtime.decode(kwargs["input_ids"][0])
            assert ("EXTERNAL_OBSERVATION_O1" in conditioning) == bool(env.responses)
            assert "TERMINAL_NOT_IN_TOKENS" not in conditioning
            action = "search[shoes]" if not env.responses else "click[buy now]"
            word = "yes" if env.member == 0 else "no"
            candidates = [runtime(f"Thought: {word} {suffix}\nAction: {action}", add_special_tokens=False)["input_ids"]
                          + [runtime.eos_token_id] for suffix in ("a", "b")]
            length = kwargs["input_ids"].shape[1]

            def allowed(batch_id, sequence):
                prefix = sequence[length:].tolist()
                result = sorted({candidate[len(prefix)] for candidate in candidates
                                 if candidate[:len(prefix)] == prefix and len(prefix) < len(candidate)})
                assert result
                return result
            kwargs["prefix_allowed_tokens_fn"] = allowed
            return original_generate(**kwargs)

        monkeypatch.setattr(trainer.model, "generate", generate)
        original_rollout = trainer.rollout_func

        def rollout(prompts, current):
            assert current is trainer
            assert current.accelerator.unwrap_model(current.model_wrapped) is trainer.model
            assert all(prompt == [{"role": "user", "content": ""}] for prompt in prompts)
            metadata = deepcopy(current._shop_batch)
            model_hash = adapter_hash(trainer.model)
            result = original_rollout(prompts, current)
            for initial, ids, logprobs, mask in zip(result["prompt_ids"], result["completion_ids"],
                                                    result["logprobs"], result["env_mask"]):
                assert len(ids) == len(logprobs) == len(mask)
                text = runtime.decode(initial + ids)
                assert "TASK_A_PRIVATE" not in text and "TASK_B_PRIVATE" not in text
                assert "schedule_index" not in text and "PRIVATE" not in text
                assert "TERMINAL_NOT_IN_TOKENS" not in text and "EXTERNAL_OBSERVATION_O1" in text
                assert sum(token == runtime.eos_token_id and own == 1 for token, own in zip(ids, mask)) == 2
                assert ids[-1] == runtime.eos_token_id and mask[-1] == 1
                assert torch.isfinite(torch.tensor(logprobs)).all()
            batches.append({"step": current.state.global_step, "metadata": list(metadata),
                            "model_hash": model_hash, "result": result})
            return result
        monkeypatch.setattr(trainer, "rollout_func", rollout)

        original_reward = trainer.reward_funcs[0]
        def reward(**kwargs):
            calls_before = sum(len(env.responses) for env in envs)
            values = original_reward(**kwargs)
            assert calls_before == sum(len(env.responses) for env in envs)
            rewards.append(list(values))
            return values
        trainer.reward_funcs[0] = reward

        original_score = trainer._generate_and_score_completions
        def score(inputs):
            prepared = original_score(inputs)
            assert trainer._shop_batch is None
            result = batches[-1]["result"]
            assert "old_per_token_logps" not in prepared
            # Rewards [1,0] have mean .5 and unbiased std sqrt(.5).
            expected = torch.tensor([.5, -.5, .5, -.5]) / (2 ** -0.5 + 1e-4)
            assert torch.allclose(prepared["advantages"].cpu(), expected)
            for i, ids in enumerate(result["completion_ids"]):
                n = len(ids)
                attention = prepared["completion_mask"][i, :n]
                tool = prepared["tool_mask"][i, :n]
                assert attention.tolist() == [1] * n
                assert tool.tolist() == result["env_mask"][i]
                assert prepared["completion_ids"][i, :n].tolist() == ids
                assert torch.allclose(prepared["sampling_per_token_logps"][i, :n], torch.tensor(result["logprobs"][i]))
                first_eos = ids.index(runtime.eos_token_id)
                assert attention[first_eos + 1:].sum() > 0 and tool[first_eos + 1:].sum() > 0
            batches[-1]["advantages"] = prepared["advantages"].tolist()
            return prepared
        monkeypatch.setattr(trainer, "_generate_and_score_completions", score)

        original_loss = trainer._compute_loss
        original_logps = trainer._get_per_token_logps_and_entropies
        current_masks = []
        def logps(*args, **kwargs):
            result = original_logps(*args, **kwargs)
            if result[0].requires_grad:
                owned = current_masks[-1].clone().bool()
                def gradient(grad):
                    assert torch.all(grad[~owned] == 0)  # Real loss-path autograd, not just our mask.
                    assert torch.any(grad[owned] != 0)
                    loss_checks.append({"external_gradient_zero": True, "model_gradient_nonzero": True})
                result[0].register_hook(gradient)
            return result
        monkeypatch.setattr(trainer, "_get_per_token_logps_and_entropies", logps)
        def loss(model, inputs):
            assert "old_per_token_logps" not in inputs
            mask = inputs["completion_mask"] * inputs["tool_mask"]
            current_masks.append(mask)
            actual = original_loss(model, inputs)
            assert torch.isfinite(actual)
            # old=None means native current.detach(): forward ratio=1, loss=-adv/GAcc.
            expected = -inputs["advantages"].mean() / trainer.current_gradient_accumulation_steps
            assert torch.allclose(actual.detach(), expected, atol=1e-6)
            current_masks.pop()
            return actual
        monkeypatch.setattr(trainer, "_compute_loss", loss)

    def construct():
        trainer = build_grpo_trainer(model=tiny_model(runtime), tokenizer=runtime, dataset=dataset,
                                     config=config, env_factory=Env, use_cpu=True)
        assert trainer.args.device.type == "cpu"
        instrument(trainer)
        return trainer

    trainer = construct()
    dataloader = trainer.get_train_dataloader()
    observed = [[row["schedule_index"] for row in batch] for batch in dataloader]
    assert observed == [[0, 0, 1, 1]] * 4 + [[2, 2, 3, 3]] * 4
    before = adapter_hash(trainer.model)
    trainer.add_callback(stop_after_one())
    train_grpo(trainer)
    after = adapter_hash(trainer.model)
    assert before != after and trainer.state.global_step == 1
    assert len(envs) == 4 and len(rewards) == 1 and len(loss_checks) == 4
    assert all(env.closed and env.released and len(env.responses) == 2 for env in envs)
    checkpoint = tmp_path / "checkpoints/checkpoint-1"
    checkpoint_states(checkpoint)

    # Fresh Trainer, identical schedule/config. Actual native checkpoint restores the adapter/state.
    prepare_run(tmp_path, config=config, inputs=identity, resume_from_checkpoint=checkpoint)
    resumed = construct()
    class Loaded(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            assert state.global_step == 1 and adapter_hash(resumed.model) == after
            assert resumed.optimizer.state and resumed.lr_scheduler.last_epoch == 1
            next_models.append({"restored_hash": adapter_hash(resumed.model), "global_step": state.global_step})
    resumed.add_callback(Loaded())
    train_grpo(resumed, resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    assert len(envs) == 8 and len(rewards) == 2 and len(loss_checks) == 8
    assert [[row["schedule_index"] for row in b["metadata"]] for b in batches] == [[0, 0, 1, 1], [2, 2, 3, 3]]
    assert batches[1]["model_hash"] == after != before
    assert "gpu_peak_allocated_bytes" not in (tmp_path / "metrics.jsonl").read_text()
    assert [env.task_id for env in envs] == ["TASK_A_PRIVATE"] * 2 + ["TASK_B_PRIVATE"] * 2 + ["TASK_A_PRIVATE"] * 2 + ["TASK_B_PRIVATE"] * 2
    assert all(env.closed and len(env.responses) == 2 for env in envs)
    report("grpo", {"dataloader_schedule_indices": observed, "reward_calls": rewards,
                    "batches": [{k: v for k, v in batch.items() if k != "result"} for batch in batches],
                    "gradient_checks": loss_checks, "before_hash": before, "after_update1_hash": after,
                    "after_update2_hash": adapter_hash(resumed.model), "resume": next_models,
                    "final_step": resumed.state.global_step, "checkpoint": str(checkpoint),
                    "env_resets": len(envs), "env_steps": sum(len(env.responses) for env in envs),
                    "old_per_token_logps": None})

    # Infrastructure error through the real parent's generation/reward path must escape.
    class Broken(Env):
        def step(self, *args, **kwargs):
            raise TeacherEnvError("fixture disconnect", kind="infrastructure")
    broken_config = deepcopy(config)
    broken_config["output_dir"] = str(tmp_path / "broken")
    (tmp_path / "broken").mkdir()
    broken = build_grpo_trainer(model=tiny_model(runtime), tokenizer=runtime, dataset=dataset,
                                config=broken_config, env_factory=Broken, use_cpu=True)
    instrument(broken)
    with pytest.raises(TeacherEnvError, match="fixture disconnect"):
        broken._generate_and_score_completions(next(iter(broken.get_train_dataloader())))
    assert broken._shop_batch is None and envs[-1].closed and envs[-1].released


def test_runtime_reward_cli_and_callback():
    from training.grpo import rollout_reward
    from training.rollout import AgentRollout
    from tests.training.test_rollout import FakeEnv, FakePolicy, step_payload
    parse = runpy.run_path(str(ROOT / "scripts/train_grpo.py"))["parse_config"]
    base = ["--scenario", "single", "--model-path", "unused", "--train-task-manifest", "unused",
            "--output-dir", "unused", "--per-device-train-batch-size", "1", "--lora-r", "2",
            "--lora-alpha", "4", "--target-modules", "q_proj"]
    for flags, alpha, expected in [(["--reward", "strict"], 1, .5), (["--reward", "loose"], 0, .8),
                                   (["--alpha", ".5"], .5, .65)]:
        config, _, _ = parse(base + flags)
        assert config["reward_alpha"] == alpha
        terminal = step_payload("done", done=True)
        terminal.update(reward=.8, reward_detail={"r_type": 1, "r_att": .5, "r_option": 1, "r_price": 1})
        env = FakeEnv([terminal])
        result = AgentRollout(policy=FakePolicy(["Thought: OK\nAction: click[buy now]"]), env_factory=lambda: env,
                               scenario="single", reward_alpha=alpha).run("t")
        assert rollout_reward(rollout_reward=[result.reward]) == pytest.approx([expected])
        assert len(env.responses) == 1
    with pytest.raises(SystemExit):
        parse(base + ["--reward", "strict", "--alpha", ".5"])
