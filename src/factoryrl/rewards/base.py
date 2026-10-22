"""Reward modes. Constraints stay hard; the reward never legalizes an action."""

from __future__ import annotations

REWARD_NEGATIVE_ELAPSED = "negative_elapsed_time"
REWARD_TERMINAL_MAKESPAN = "terminal_makespan"
REWARD_MODES = (REWARD_NEGATIVE_ELAPSED, REWARD_TERMINAL_MAKESPAN)
