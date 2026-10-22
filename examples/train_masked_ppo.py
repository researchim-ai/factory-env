"""Optional MaskablePPO smoke. The dependency is not part of the base install.

pip install "factoryrl[rl]"
python examples/train_masked_ppo.py
"""

from __future__ import annotations

import argparse

from factoryrl.environments.job_shop import JobShopEnv
from factoryrl.scenarios.generator import generate_job_shop


def main() -> None:
    parser = argparse.ArgumentParser(description="Short MaskablePPO run on a tiny job shop.")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("sb3-contrib is not installed. Install it with: pip install 'factoryrl[rl]'")
        print("The base package stays free of PyTorch. See examples/train_masked_policy.py.")
        return
    instance = generate_job_shop(2, 2, seed=args.seed, low=1, high=6)
    env = JobShopEnv(instance=instance, check_invariants=False, record_events=False)
    model = MaskablePPO("MultiInputPolicy", env, seed=args.seed, verbose=0)
    model.learn(total_timesteps=args.timesteps)
    observation, info = env.reset(seed=args.seed)
    while True:
        action, _states = model.predict(observation, action_masks=info["action_mask"], deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print({"success": info["success"], "makespan": info["episode_metrics"]["makespan"]})


if __name__ == "__main__":
    main()
