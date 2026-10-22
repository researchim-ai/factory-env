"""Tiny masked policy gradient. It learns nothing claimed; it shows the Gymnasium loop.

Any legal dispatch policy finishes a v0.1 job shop, so a successful eval only
proves that the mask was respected.
"""

from __future__ import annotations

import argparse

import numpy as np

from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.observations.builders import N_OPERATION_FEATURES
from factoryrl.scenarios.generator import generate_job_shop


def train(*, episodes: int = 20, seed: int = 0, learning_rate: float = 0.05) -> dict[str, float | bool]:
    """Train a linear masked policy with REINFORCE for a few episodes."""
    rng = np.random.default_rng(seed)
    instance = generate_job_shop(2, 2, seed=seed, low=1, high=6)
    env = JobShopEnv(instance=instance, check_invariants=True)
    weights = rng.normal(scale=0.1, size=N_OPERATION_FEATURES)

    def policy(observation: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = observation["operations"].astype(np.float64)
        legal = np.asarray(mask[:-1], dtype=bool)
        logits = features @ weights
        logits = np.where(legal, logits, -1e9)
        shifted = logits - np.max(logits)
        probabilities = np.exp(np.where(legal, shifted, -1e9))
        probabilities /= probabilities.sum()
        return features, probabilities

    def scores(observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["operations"][:, :N_OPERATION_FEATURES] @ weights

    for _episode in range(episodes):
        observation, info = env.reset(seed=seed)
        gradients: list[np.ndarray] = []
        rewards: list[float] = []
        while True:
            features, probabilities = policy(observation, info["action_mask"])
            action = int(rng.choice(len(probabilities), p=probabilities))
            gradients.append(features[action] - probabilities @ features)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        returns = np.cumsum(rewards[::-1])[::-1]
        advantage = returns - float(returns.mean())
        for gradient, gain in zip(gradients, advantage, strict=True):
            weights += learning_rate * float(gain) * gradient

    observation, info = env.reset(seed=seed + 1)
    while True:
        logits = scores(observation)
        logits = np.where(info["action_mask"][:-1], logits, -1e9)
        action = int(np.argmax(logits))
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    makespan = info["episode_metrics"]["makespan"]
    return {
        "success": bool(info["success"]),
        "makespan": -1.0 if makespan is None else float(makespan),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-train a masked linear policy.")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = train(episodes=args.episodes, seed=args.seed)
    print(result)


if __name__ == "__main__":
    main()
