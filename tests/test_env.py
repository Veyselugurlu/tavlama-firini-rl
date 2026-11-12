import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from furnace_rl.baselines.pid import PIDPolicy
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.evaluate import run_episode
from furnace_rl.metrics import episode_metrics

NO_BATCH: list[dict] = []
ONE_BATCH = [{"t_arrival": 300.0, "mass": 1000.0, "speed": 0.2}]


class FixedPolicy:
    """Her adımda aynı güç oranını (skaler ya da bölge başına) uygular."""

    def __init__(self, u):
        self.a = (2.0 * np.broadcast_to(np.asarray(u, float), (3,)) - 1.0).astype(np.float32)

    def predict(self, obs, deterministic=True):
        return self.a, None


def short_env(**cfg):
    """FixedPolicy mutlak güç ürettiği için testler mutlak eylem modunda."""
    return FurnaceEnv({"episode_steps": 120, "action_mode": "absolute", **cfg})


def test_check_env():
    check_env(FurnaceEnv(), skip_render_check=True)


def test_seed_reproducible():
    env = FurnaceEnv()
    a = run_episode(env, PIDPolicy(env), seed=7)
    b = run_episode(env, PIDPolicy(env), seed=7)
    np.testing.assert_array_equal(a["T_meas"], b["T_meas"])
    assert a["arrivals"] == b["arrivals"]


def test_zero_power_cools_full_power_heats():
    env = short_env(noise_std=0.0)
    cold = run_episode(env, FixedPolicy(0.0), NO_BATCH, seed=0)
    hot = run_episode(env, FixedPolicy(1.0), NO_BATCH, seed=0)
    T0 = env.T_ref
    assert np.all(cold["T_true"][-1] < T0 - 5)
    assert np.all(hot["T_true"][-1] > T0 + 5)


def test_steady_state_holds_without_load():
    env = short_env(noise_std=0.0, T_init_jitter=0.0)
    env.reset(seed=0, options={"scenario": NO_BATCH})
    u = env.steady_state_power()
    tr = run_episode(env, FixedPolicy(u), NO_BATCH, seed=0)
    assert np.max(np.abs(tr["T_true"][-1] - env.T_ref)) < 1.0


def test_batch_cools_entry_zone():
    env = short_env(noise_std=0.0, T_init_jitter=0.0)
    env.reset(seed=0, options={"scenario": NO_BATCH})
    hold = FixedPolicy(env.steady_state_power())
    base = run_episode(env, hold, NO_BATCH, seed=0)
    loaded = run_episode(env, hold, ONE_BATCH, seed=0)
    assert loaded["T_true"][-1, 0] < base["T_true"][-1, 0] - 5


def test_energy_consistent():
    env = short_env()
    tr = run_episode(env, FixedPolicy(0.5), ONE_BATCH, seed=0)
    assert np.all(tr["energy_kwh"] >= 0)
    # u_eff → 0.5'e yaklaşırken toplam enerji tam güç enerjisinin yarısına yakın olmalı
    full = env.P_max.sum() * env.dt * len(tr["t"]) / 3.6e6
    assert 0.3 * full < tr["energy_kwh"].sum() < 0.6 * full


def test_safety_violation_penalized_not_terminated():
    env = short_env(noise_std=0.0)
    tr = run_episode(env, FixedPolicy(1.0), NO_BATCH, seed=0, options={"T_init": [940.0, 940.0, 940.0]})
    assert len(tr["t"]) == 120 and not tr["terminated"]       # bölüm erken bitmez
    m = episode_metrics(tr, env.dt)
    assert m["unsafe_min"] > 0 and m["return"] < -2.0 * 0.5 * 120


def test_incremental_action_mode():
    env = FurnaceEnv({"episode_steps": 5})
    env.reset(seed=0, options={"scenario": NO_BATCH})
    u0 = env.u_prev.copy()
    _, _, _, _, info = env.step(np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(info["u_cmd"], np.clip(u0 + 0.1, 0, 1))
    np.testing.assert_allclose(env.power_to_action(u0 + 0.05, u0), 0.5, atol=1e-6)


def test_obs_contains_actuator_state():
    env = FurnaceEnv()
    obs, _ = env.reset(seed=0, options={"scenario": NO_BATCH})
    assert obs.shape == (6 * 3 + 3 + 3 * 3,)
    obs, *_ = env.step(np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(obs[env.obs_slices["u_eff"]], env.u_eff, atol=1e-6)
    # en son komut bekleyen kuyruğun sonunda
    np.testing.assert_allclose(obs[env.obs_slices["pending"]][-3:], env.u_prev, atol=1e-6)


def test_preview_flag():
    batch = [{"t_arrival": 120.0, "mass": 900.0, "speed": 0.2}]
    on, off = FurnaceEnv(), FurnaceEnv({"use_preview": False})
    o_on, _ = on.reset(seed=0, options={"scenario": batch})
    o_off, _ = off.reset(seed=0, options={"scenario": batch})
    sl = on.obs_slices["preview"]
    assert o_on[sl][0] == pytest.approx(900.0 / 1500.0)
    np.testing.assert_array_equal(o_off[sl], [0.0, 0.0, 1.0])


def test_metrics_shapes():
    env = short_env()
    m = episode_metrics(run_episode(env, PIDPolicy(env), ONE_BATCH, seed=0), env.dt)
    assert m["n_batches"] == 1 and m["rmse"] >= 0 and m["energy_kwh"] > 0
