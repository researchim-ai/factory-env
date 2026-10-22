"""Independent NumPy random streams.

The v0.1 job shop does not use randomness inside the transition function.
Streams exist so generators and baselines do not share state with future
stochastic failures.
"""

from __future__ import annotations

import numpy as np

from factoryrl.exceptions import ConfigurationError

RNG_ALGORITHM = "numpy.random.PCG64"


def make_generator(seed: int | None) -> np.random.Generator:
    """Build a PCG64 generator. ``None`` draws entropy from the OS."""
    if seed is not None:
        validate_seed(seed)
    return np.random.default_rng(seed)


def spawn_generators(seed: int, names: tuple[str, ...]) -> dict[str, np.random.Generator]:
    """Split one seed into independent generators, one per subsystem name."""
    validate_seed(seed)
    sequence = np.random.SeedSequence(seed)
    children = sequence.spawn(len(names))
    return {
        name: np.random.Generator(np.random.PCG64(child)) for name, child in zip(names, children, strict=True)
    }


def validate_seed(seed: int) -> None:
    """Reject booleans, negatives, and non-integers."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError(
            "seed must be a non-negative integer.",
            context={"seed": seed},
        )
