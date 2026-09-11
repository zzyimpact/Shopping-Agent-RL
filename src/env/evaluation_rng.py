"""TEST-only RNG scopes around unchanged upstream random calls."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import random


ENVIRONMENT_RNG_STRATEGY = "test_runtime_task_session_sha256_v1"
_current_rng = ContextVar("shopsim_test_rng", default=None)


def rng_contract(base_seed):
    if type(base_seed) is not int or not 0 <= base_seed < 2**32:
        raise ValueError("formal_eval_seed must be an integer in [0, 2**32)")
    return {"strategy": ENVIRONMENT_RNG_STRATEGY, "formal_eval_seed": base_seed}


def local_rng(base_seed, scope, scenario=None, task_id=None):
    contract = rng_contract(base_seed)
    identity = [contract["strategy"], base_seed, scope, scenario, task_id]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


class _ScopedRandom:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        rng = _current_rng.get()
        return getattr(self.original if rng is None else rng, name)


@contextmanager
def upstream_rng_scope(rng, modules):
    # Replace only upstream modules' `random` references, never random.seed/state.
    # ContextVar isolates threads/nested goal generation; outside TEST scopes the
    # original random module remains the source, including the unchanged TRAIN path.
    for module in modules:
        if not isinstance(module.random, _ScopedRandom):
            module.random = _ScopedRandom(module.random)
    token = _current_rng.set(rng)
    try:
        yield
    finally:
        _current_rng.reset(token)
