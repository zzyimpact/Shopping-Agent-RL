"""Append-only token ownership for one ShopSimulator trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field

from training.policy import GenerationConfig, PolicySample


@dataclass
class TokenTrace:
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    env_mask: list[int] = field(default_factory=list)

    def add(self, ids, logprobs, *, generated: bool) -> None:
        if len(ids) != len(logprobs):
            raise ValueError("token/logprob alignment mismatch")
        self.completion_ids.extend(ids)
        self.logprobs.extend(logprobs)
        self.env_mask.extend([int(generated)] * len(ids))

    def sample_turn(self, policy, messages, sampling: GenerationConfig) -> PolicySample | None:
        if not self.prompt_ids:
            self.prompt_ids = policy.prompt_token_ids(messages, sampling)
            inserted = []
        else:
            inserted = policy.observation_token_ids(
                messages[-1]["content"], previous_token=self.completion_ids[-1], sampling=sampling,
            )
        prefix = self.prompt_ids + self.completion_ids + inserted
        if len(prefix) >= sampling.max_context_tokens:
            if not self.completion_ids:
                raise ValueError("initial prompt exceeds context budget")
            return None  # End episode; never append a trailing observation with no next sample.
        sample = policy.sample(prefix, sampling=sampling)
        if len(prefix) + len(sample.token_ids) > sampling.max_context_tokens:
            raise ValueError("sampling backend exceeded context budget")
        self.add(inserted, [0.0] * len(inserted), generated=False)
        self.add(sample.token_ids, sample.logprobs, generated=True)
        return sample
