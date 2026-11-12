"""E2 – öğrenme eğrileri: doğrulama senaryolarında ödül ve RMSE (seed ortalaması ± std).

Örnek:
    python scripts/plot_learning_curves.py --runs results/models/q_table_seed0 results/models/q_table_seed1 results/models/q_table_seed2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.baselines.pid import PIDPolicy
from furnace_rl.config import load_env_config, load_yaml
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import make_test_set
from furnace_rl.evaluate import evaluate
from furnace_rl.plots import plot_learning_curves


def pid_reference(cfg: dict, env_config: str | None) -> pd.Series:
    """PID'in doğrulama metrikleri; ödül, RL koşularının eğitildiği ağırlıklarla hesaplanır."""
    env = FurnaceEnv(load_env_config(env_config))
    val = make_test_set(env.cfg, n=cfg["n_val_scenarios"], base_seed=cfg["val_base_seed"])
    df, _ = evaluate(PIDPolicy.from_yaml(env, "configs/pid.yaml"), env, val)
    return df.mean(numeric_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--label", default="Q-tablosu")
    ap.add_argument("--out", default="results/figures/e2_learning")
    ap.add_argument("--config", default="configs/q_table.yaml", help="doğrulama senaryolarını veren Q-tablosu ayarı")
    ap.add_argument("--env-config", help="koşuların eğitildiği ortam ayarı (PID referansı aynı ödülle ölçülür)")
    ap.add_argument("--min-step", type=int, default=0, help="bu adımdan önceki noktaları çizme")
    args = ap.parse_args()

    logs = [pd.read_csv(Path(r) / "evaluations.csv") for r in args.runs]
    logs = [l[l["step"] >= args.min_step].reset_index(drop=True) for l in logs]
    n = min(len(l) for l in logs)
    steps = logs[0]["step"].to_numpy()[:n]
    pid = pid_reference(load_yaml(args.config), args.env_config)

    for key, ylabel in [("return", "bölüm ödülü (doğrulama)"), ("rmse", "RMSE [°C] (doğrulama)"),
                        ("energy_kwh", "enerji [kWh] (doğrulama)")]:
        Y = np.stack([l[key].to_numpy()[:n] for l in logs])
        plot_learning_curves({args.label: (steps, Y)}, f"{args.out}_{key}.png", ylabel, reference=("PID", pid[key]))
        print(f"{key}: son {Y[:, -1].mean():.2f} ± {Y[:, -1].std():.2f}   en iyi {Y.max(1).mean() if key == 'return' else Y.min(1).mean():.2f}   PID {pid[key]:.2f}")


if __name__ == "__main__":
    main()
