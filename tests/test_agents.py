import re
from pathlib import Path

import numpy as np

from furnace_rl.agents.q_table import QTablePolicy, StateCoder, load_class, to_env_action, zone_rewards
from furnace_rl.config import load_yaml
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.evaluate import run_episode

ROOT = Path(__file__).resolve().parents[1]
CFG = load_yaml(ROOT / "configs/q_table.yaml")
BATCH = [{"t_arrival": 120.0, "mass": 1000.0, "speed": 0.2}]


def test_no_neural_network_code():
    """Ödev şartı: derin RL yok. Kodda ve bağımlılıklarda sinir ağı kütüphanesi olmamalı."""
    banned = re.compile(r"^\s*(import|from)\s+(torch|tensorflow|keras|jax|flax|stable_baselines3|sb3_contrib)\b", re.M)
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]:
        assert not banned.search(path.read_text(encoding="utf-8")), path
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for name in ("torch", "tensorflow", "keras", "stable-baselines3"):
        assert name not in req


def test_grid_is_10_by_10_and_state_index_in_range():
    env = FurnaceEnv()
    coder = StateCoder(CFG, env)
    assert coder.shape == (10, 10) and coder.n_states == 100
    obs, _ = env.reset(seed=0, options={"scenario": BATCH})
    s = coder(obs)
    assert s.shape == (3,) and s.min() >= 0 and s.max() < coder.n_states


def test_load_classes_cover_all_ten_cases():
    b = CFG["bins"]
    rate = np.array([-9, 0, 9, 0, 0, 0, 9, 0, 9, 0], float)
    load = np.array([0, 0, 0, 0, 0, 0.3, 0.3, 0.8, 0.8, 0.8])
    coming = np.array([0, 0, 0, 0.2, 0.8, 0, 0, 0, 0, 0.8])
    assert load_class(rate, load, coming, b).tolist() == list(range(10))


def test_env_action_respects_rate_limit():
    env = FurnaceEnv()
    obs, _ = env.reset(seed=0)
    a = to_env_action(np.array([1.0, 0.0, 0.35]), obs, env.obs_slices, env.cfg["du_max"])
    assert a.min() >= -1.0 and a.max() <= 1.0 and a.dtype == np.float32


def test_policy_holds_level_between_decisions_and_episode_runs():
    env = FurnaceEnv({"episode_steps": 30})
    coder = StateCoder(CFG, env)
    levels = np.asarray(CFG["levels"], float)
    q = np.random.default_rng(0).normal(size=(3, coder.n_states, len(levels)))
    policy = QTablePolicy(q, coder, levels, k=CFG["decision_steps"], du_max=env.cfg["du_max"])
    traj = run_episode(env, policy, BATCH, seed=0)
    assert len(traj["t"]) == 30
    assert traj["u_cmd"].min() >= 0.0 and traj["u_cmd"].max() <= 1.0
    policy.reset()
    obs, _ = env.reset(seed=0)
    policy.predict(obs)
    target = policy.target.copy()
    for _ in range(CFG["decision_steps"] - 1):
        policy.predict(obs)
        assert np.array_equal(policy.target, target)


def test_zone_rewards_sum_to_env_reward_without_safety():
    env = FurnaceEnv()
    obs, _ = env.reset(seed=0, options={"scenario": BATCH})
    for _ in range(20):
        env._u_before = env.u_prev.copy()
        obs, r, _, _, info = env.step(np.full(3, 0.5, dtype=np.float32))
        assert np.isclose(zone_rewards(info, env).sum(), r, atol=1e-6)


def test_q_update_moves_value_toward_target():
    """Slayttaki güncelleme: Q(s,a) ← Q(s,a) + α [r + γ max Q(s',·) − Q(s,a)]."""
    q = np.zeros((1, 2, 2))
    r, gamma, alpha = -1.0, 0.9, 0.5
    q[0, 1] = [-2.0, -4.0]
    td = r + gamma * q[0, 1].max() - q[0, 0, 0]
    q[0, 0, 0] += alpha * td
    assert np.isclose(q[0, 0, 0], 0.5 * (-1.0 + 0.9 * -2.0))
