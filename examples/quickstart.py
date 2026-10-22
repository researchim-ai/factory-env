"""Run the specification's two-job example and print makespan and return."""

from __future__ import annotations

from factoryrl import JobShopEnv, SPTAgent, demo_instance


def main() -> None:
    env = JobShopEnv(instance=demo_instance(), invalid_action_mode="raise")
    agent = SPTAgent()
    observation, info = env.reset(seed=42)
    agent.reset(seed=42)
    total = 0.0
    while True:
        action = agent.act(observation, info["action_mask"])
        observation, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    metrics = info["episode_metrics"]
    print(f"makespan={metrics['makespan']} return={total} success={metrics['success']}")
    print("This checks one legal schedule. It is not a claim of global optimality.")


if __name__ == "__main__":
    main()
