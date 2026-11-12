"""E5 – hiperparametre analizi: her ayar 3 seed ile, 20 test senaryosunda ve doğrulama eğrilerinde.

Son iki satır en iyi iki bulgunun birleşimidir; son ayar doğrulama RMSE'sine göre seçildi."""

from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import load_scenarios
from furnace_rl.evaluate import evaluate, make_policy
from furnace_rl.plots import plot_learning_curves, to_markdown

RUNS = {
    "referans (10×10 ızgara, α_min=0.02, ε 0.3→0.02)": "q_table_seed{}",
    "güç seviyeleri 0/0.2/0.3/0.4/0.6/1": "e5q_levels_seed{}",
    "α_min = 0.05": "e5q_alphamin_seed{}",
    "ε 0.5 → 0.05": "e5q_eps_seed{}",
    "ızgara: hata × değişim hızı (yük yok)": "e5q_errorrate_seed{}",
    "seviyeler 0/0.2/0.3/0.4/0.6/1 + w₂=1.0": "e5q_levels_w1_seed{}",
    "son ayar: seviyeler + w₂=1.0 + 30 s karar": "q_son_seed{}",
}
# w₂=1.0 ile eğitilenlerin doğrulama ödülü başka ölçekte: eğri grafiğine girmez.
# Test tablosu herkes için varsayılan ödülle (w₂=0.2) ölçülür.
OTHER_REWARD = {"e5q_levels_w1_seed{}", "q_son_seed{}"}


def main():
    env = FurnaceEnv()
    scenarios = load_scenarios("configs/test_scenarios.json")
    rows, curves = [], {}
    for label, pattern in RUNS.items():
        logs = [pd.read_csv(Path("results/models") / pattern.format(s) / "evaluations.csv") for s in range(3)]
        if pattern not in OTHER_REWARD:
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
