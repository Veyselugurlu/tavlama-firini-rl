"""Politikaları aynı senaryolarda koşturup metrik tablosu üretir.

Örnek:
    python -m furnace_rl.evaluate --policy pid --pid-config configs/pid.yaml --out results/eval/pid
    python -m furnace_rl.evaluate --policy rl --model results/models/q_table_seed0 --out results/eval/q_table_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.config import load_env_config
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import load_scenarios
from furnace_rl.metrics import episode_metrics

TRAJ_KEYS = ("t", "T_true", "T_meas", "T_ref", "u_cmd", "u_eff", "energy_kwh", "reward", "zone_load")


def run_episode(env: FurnaceEnv, policy, scenario: list[dict] | None = None, seed: int | None = None,
                options: dict | None = None) -> dict:
    options = dict(options or {})
    if scenario is not None:
        options["scenario"] = scenario
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "reset"):
        policy.reset()
    rows = {k: [] for k in TRAJ_KEYS}
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        info["reward"] = reward
        for k in TRAJ_KEYS:
            rows[k].append(info[k])
    traj = {k: np.asarray(v) for k, v in rows.items()}
    traj["arrivals"] = info["arrivals"]
    traj["terminated"] = terminated
    return traj


def evaluate(policy, env: FurnaceEnv, scenarios: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    records, trajs = [], []
    for sc in scenarios:
        traj = run_episode(env, policy, sc["schedule"], sc["seed"])
        m = episode_metrics(traj, env.dt)
        m["scenario"] = sc["name"]
        records.append(m)
        trajs.append(traj)
    df = pd.DataFrame(records).set_index("scenario")
    return df, trajs


def load_run_policy(run_dir: str | Path, env: FurnaceEnv):
    """Eğitim klasöründen en iyi Q-tablosunu yükler. Ölçüm `env` üzerinde yapılır."""
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    best = run_dir / "best_model.npz"
    if "q_table" in saved:
        from furnace_rl.agents.q_table import QTablePolicy, StateCoder
        z = np.load(best)
        cfg_env = FurnaceEnv(saved["env"])
        return QTablePolicy(z["q"], StateCoder(saved["q_table"], cfg_env), z["levels"], int(z["k"]), cfg_env.cfg["du_max"])
    raise ValueError(f"bilinmeyen eğitim klasörü: {run_dir}")


def make_policy(kind: str, env: FurnaceEnv, pid_config: str | None = None, model: str | None = None):
    if kind == "pid":
        from furnace_rl.baselines.pid import PIDPolicy
        return PIDPolicy.from_yaml(env, pid_config) if pid_config else PIDPolicy(env)
    if kind == "rl":
        return load_run_policy(model, env)
    raise ValueError(f"bilinmeyen politika: {kind}")


def save_trajs(trajs: list[dict], names: list[str], out_dir: Path) -> None:
    d = out_dir / "trajs"
    d.mkdir(parents=True, exist_ok=True)
    for name, tr in zip(names, trajs):
        np.savez_compressed(d / f"{name}.npz", arrivals=np.asarray(tr["arrivals"]),
                            terminated=tr["terminated"], **{k: tr[k] for k in TRAJ_KEYS})


def load_traj(path: str | Path) -> dict:
    z = np.load(path)
    traj = {k: z[k] for k in TRAJ_KEYS}
    traj["arrivals"] = list(z["arrivals"])
    traj["terminated"] = bool(z["terminated"])
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["pid", "rl"], required=True)
    ap.add_argument("--pid-config", default="configs/pid.yaml")
    ap.add_argument("--model", help="eğitim klasörü, ör. results/models/q_table_seed0")
    ap.add_argument("--env-config")
    ap.add_argument("--noise-std", type=float, help="sensör gürültüsünü üzerine yaz (E6)")
    ap.add_argument("--scenarios", default="configs/test_scenarios.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    overrides = {} if args.noise_std is None else {"noise_std": args.noise_std}
    env = FurnaceEnv(load_env_config(args.env_config, **overrides))
    policy = make_policy(args.policy, env, args.pid_config, args.model)
    scenarios = load_scenarios(args.scenarios)
    df, trajs = evaluate(policy, env, scenarios)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv")
    save_trajs(trajs, list(df.index), out)
    summary = df.drop(columns=["n_batches"]).agg(["mean", "std"]).T
    summary.to_csv(out / "summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
