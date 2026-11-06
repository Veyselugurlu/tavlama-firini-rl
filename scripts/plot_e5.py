"""E5 – hiperparametre analizi: her ayar 3 seed ile, 20 test senaryosunda ve doğrulama eğrilerinde."""

from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import load_scenarios
from furnace_rl.evaluate import evaluate, make_policy
from furnace_rl.plots import plot_learning_curves, to_markdown

RUNS = {
    "referans (H=60, λ=1e-3, N=200, ρ=0.995)": "mbrl_seed{}",
    "ufuk H=30 (5 dk)": "e5_horizon30_seed{}",
    "ufuk H=120 (20 dk)": "e5_horizon120_seed{}",
    "yeniden plan 30 s": "e5_replan3_seed{}",
    "CEM nüfusu 50": "e5_cem_pop50_seed{}",
    "bozucu kestirimi yok": "e5_bias_filter0.0_seed{}",
    "taze yük ρ=0.99": "e5_fresh_rho0.99_seed{}",
    "taze yük ρ=0.999": "e5_fresh_rho0.999_seed{}",
    "ridge λ=1": "e5_lam1.0_seed{}",
}


def main():
    env = FurnaceEnv()
    scenarios = load_scenarios("configs/test_scenarios.json")
    rows, curves = [], {}
    for label, pattern in RUNS.items():
        logs = [pd.read_csv(Path("results/models") / pattern.format(s) / "evaluations.csv") for s in range(3)]
        curves[label] = (logs[0].step.to_numpy(), np.stack([l["return"].to_numpy() for l in logs]))
        test = [evaluate(make_policy("rl", env, model=f"results/models/{pattern.format(s)}"), env, scenarios)[0]
                for s in range(3)]
        means = pd.DataFrame([t.mean(numeric_only=True) for t in test])
        rows.append({"ayar": label, "test ödül": round(means["return"].mean(), 1),
                     "ödül std": round(means["return"].std(), 1), "test RMSE": round(means.rmse.mean(), 2),
                     "RMSE std": round(means.rmse.std(), 2), "test düşüş": round(means.max_drop.mean(), 1),
                     "test aşma": round(means.overshoot.mean(), 1)})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).set_index("ayar")
    table.to_csv("results/eval/e5_hparams.csv")
    Path("results/eval/e5_hparams.md").write_text(to_markdown(table), encoding="utf-8")
    print(table.to_string())
    plot_learning_curves(curves, "results/figures/e5_return.png", "bölüm ödülü (doğrulama)")


if __name__ == "__main__":
    main()
