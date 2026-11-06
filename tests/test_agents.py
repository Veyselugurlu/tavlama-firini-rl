import re
from pathlib import Path

import numpy as np

from furnace_rl.agents import ars
from furnace_rl.agents.mbrl import LinearModel, LoadTracker, MPCPolicy
from furnace_rl.agents.q_tile import TileCoder, TileQPolicy, zone_states
from furnace_rl.config import load_yaml
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.evaluate import run_episode

ROOT = Path(__file__).resolve().parents[1]
BATCH = [{"t_arrival": 120.0, "mass": 1000.0, "speed": 0.2}]


def test_no_neural_network_code():
    """Ödev şartı: derin RL yok. Kodda ve bağımlılıklarda sinir ağı kütüphanesi olmamalı."""
    banned = re.compile(r"^\s*(import|from)\s+(torch|tensorflow|keras|jax|flax|stable_baselines3|sb3_contrib)\b", re.M)
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]:
        assert not banned.search(path.read_text(encoding="utf-8")), path
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for name in ("torch", "tensorflow", "keras", "stable-baselines3"):
        assert name not in req


def test_tile_coder_indices_in_range_and_generalize():
    coder = TileCoder([-1.0, 0.0], [1.0, 1.0], [5, 4], n_tilings=8)
    near_a, near_b = coder.tiles(np.array([0.1, 0.5])), coder.tiles(np.array([0.11, 0.5]))
    far = coder.tiles(np.array([-1.0, 0.0]))
    assert near_a.shape == (8,) and near_a.min() >= 0 and near_a.max() < coder.size
    assert len(set(near_a) & set(near_b)) > len(set(near_a) & set(far))    # yakın durumlar kare paylaşır


def test_tile_q_policy_holds_action_between_decisions():
    env = FurnaceEnv()
    coder = TileCoder([-1.6, -1, -0.5, 0, 0], [1.6, 1, 0.5, 1.2, 1], [5, 3, 3, 2, 2], n_tilings=2)
    w = np.random.default_rng(0).normal(size=(3, coder.size, 3))
    policy = TileQPolicy(w, coder, np.array([0.0, -0.5, 0.5]), env.obs_slices, decision_steps=3)
    obs, _ = env.reset(seed=0)
    first = policy.predict(obs)[0]
    assert zone_states(obs, env.obs_slices).shape == (3, 5)
    assert np.array_equal(first, policy.predict(obs)[0]) and np.array_equal(first, policy.predict(obs)[0])


def test_ars_features_are_zero_at_rest_so_no_steady_offset():
    env = FurnaceEnv({"noise_std": 0.0, "T_init_jitter": 0.0})
    obs, _ = env.reset(seed=0, options={"scenario": []})
    z = ars.state_features(obs, env.obs_slices)
    assert z.shape == (1, ars.N_FEATURES)
    assert np.allclose(z, 0.0, atol=1e-5)


def test_linear_model_recovers_known_dynamics():
    rng = np.random.default_rng(0)
    N = 5000
    e, u, l = rng.normal(size=(N, 3)), rng.random((N, 3)), rng.random((N, 6))
    W = rng.normal(0, 0.1, (13, 3))
    X = np.hstack([e, u, l, np.ones((N, 1))])
    u_del = rng.random((N, 3))
    data = {"e": e, "u_eff": u, "l": l, "e_next": e + X @ W, "u_del": u_del, "u_eff_next": 0.9 * u + 0.1 * u_del}
    model = LinearModel(3, lam=1e-6)
    model.fit(data)
    assert np.allclose(model.W_e, W, atol=1e-3)
    assert np.allclose(model.alpha, 0.9, atol=1e-3) and np.allclose(model.beta, 0.1, atol=1e-3)


def test_load_tracker_marks_only_new_mass_as_fresh():
    tr = LoadTracker(3, rho=0.5)
    assert np.allclose(tr.update(np.array([1.0, 0, 0]))[3:], [1.0, 0, 0])
    assert np.allclose(tr.update(np.array([1.0, 0, 0]))[3:], [0.5, 0, 0])
    assert np.allclose(tr.update(np.array([0.0, 1.0, 0]))[3:], [0.25, 1.0, 0])   # sonraki bölgeye geçiş


def _mpc(env: FurnaceEnv) -> MPCPolicy:
    # Isıtma etkisi olan basit bir model: hatayı azaltmak için gücü artırmak işe yarar.
    model = LinearModel(3, 0.0)
    model.W_e = np.zeros((13, 3))
    model.W_e[3:6, :] = np.eye(3) * 0.01
    model.alpha, model.beta = np.full(3, 0.9), np.full(3, 0.1)
    cfg = {**load_yaml(ROOT / "configs/mbrl.yaml"), "cem_pop": 30, "cem_iters": 2}
    return MPCPolicy(model, env, cfg, seed=0)


def test_mpc_actions_are_bounded_and_episode_runs():
    env = FurnaceEnv({"episode_steps": 30})
    traj = run_episode(env, _mpc(env), BATCH, seed=0)
    assert len(traj["t"]) == 30
    assert traj["u_cmd"].min() >= 0.0 and traj["u_cmd"].max() <= 1.0


def test_mpc_is_reproducible_for_same_scenario():
    env = FurnaceEnv({"episode_steps": 30})
    policy = _mpc(env)
    a = run_episode(env, policy, BATCH, seed=0)
    b = run_episode(env, policy, BATCH, seed=0)
    assert np.array_equal(a["u_cmd"], b["u_cmd"])


def test_mpc_load_forecast_adds_previewed_batch_at_arrival():
    env = FurnaceEnv()
    policy = _mpc(env)
    obs, _ = env.reset(seed=0, options={"scenario": BATCH})
    o = policy._observe(obs)
    L = policy._load_forecast(o)
    k = int(BATCH[0]["t_arrival"] / env.dt)
    assert L.shape == (policy.H, 6)
    assert np.isclose(L[k - 1, 0], 0.0) and np.isclose(L[k, 0], BATCH[0]["mass"] / env.cfg["obs"]["mass_scale"])
