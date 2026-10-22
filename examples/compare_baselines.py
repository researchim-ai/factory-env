"""Compare FIFO and SPT on the same instances. Numbers are measurements, not a ranking claim."""

from __future__ import annotations

from factoryrl import FIFOAgent, JobShopEnv, SPTAgent, choice_instance, demo_instance


def run(instance, agent) -> float:
    env = JobShopEnv(instance=instance)
    observation, info = env.reset(seed=0)
    agent.reset(seed=0)
    while True:
        action = agent.act(observation, info["action_mask"])
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            makespan = info["episode_metrics"]["makespan"]
            assert makespan is not None
            return float(makespan)


def main() -> None:
    rows = [
        ("demo", demo_instance()),
        ("choice", choice_instance()),
    ]
    print(f"{'instance':<10} {'fifo':>8} {'spt':>8}")
    for name, instance in rows:
        fifo = run(instance, FIFOAgent())
        spt = run(instance, SPTAgent())
        print(f"{name:<10} {fifo:8.1f} {spt:8.1f}")


if __name__ == "__main__":
    main()
