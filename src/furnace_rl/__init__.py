"""Ödev 9 – çok bölgeli tavlama fırını için RL ortamı, baseline ve deney araçları."""

from gymnasium.envs.registration import register

register(id="Furnace-v0", entry_point="furnace_rl.envs.furnace_env:FurnaceEnv")
