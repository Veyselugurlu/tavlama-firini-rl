"""Yöntem karşılaştırması: tüm sinir ağısız RL yöntemleri ve PID, aynı 20 test senaryosunda (3 seed).

Çıktı: results/eval/methods.csv/.md, results/figures/methods_learning.png (doğrulama RMSE'si – etkileşim adımı,
logaritmik eksen; adım sayıları yöntemler arasında çok farklıdır).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from furnace_rl.envs.furnace_env import FurnaceEnv  # noqa: E402
from furnace_rl.envs.scenarios import load_scenarios, make_test_set  # noqa: E402
from furnace_rl.evaluate import evaluate, make_policy  # noqa: E402
from furnace_rl.plots import COLORS, to_markdown  # noqa: E402

METHODS = {"MB-RL (doğrusal model + CEM)": "mbrl_seed{}", "ARS (doğrusal politika)": "ars_seed{}",
           "Q-öğrenme (tile coding)": "q_tile_seed{}", "LSPI (karesel Q)": "lspi_seed{}"}
KEYS = ["return", "rmse", "max_drop", "overshoot", "energy_kwh"]


def main():
    env = FurnaceEnv()
    scenarios = load_scenarios("configs/test_scenarios.json")
    pid, _ = evaluate(make_policy("pid", env, "configs/pid.yaml"), env, scenarios)
    rows = [{"yöntem": "PID (baseline)", "eğitim adımı": 0, **pid[KEYS].mean().round(2).to_dict(), "RMSE std": 0.0}]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for j, (label, pattern) in enumerate(METHODS.items()):
        means, logs = [], []
        for s in range(3):
            run = f"results/models/{pattern.format(s)}"
            d, _ = evaluate(make_policy("rl", env, model=run), env, scenarios)
            means.append(d[KEYS].mean())
            logs.append(pd.read_csv(Path(run) / "evaluations.csv"))
        m = pd.DataFrame(means)
        rows.append({"yöntem": label, "eğitim adımı": int(logs[0].step.iloc[-1]), **m.mean().round(2).to_dict(),
                     "RMSE std": round(m.rmse.std(), 2)})
        print(rows[-1], flush=True)
        n = min(len(l) for l in logs)
        Y = np.stack([l.rmse.to_numpy()[:n] for l in logs])
        x = logs[0].step.to_numpy()[:n]
        ax.plot(x, Y.mean(0), color=COLORS[j % len(COLORS)], label=label, marker="o" if n < 40 else None, ms=3)
        ax.fill_between(x, Y.min(0), Y.max(0), color=COLORS[j % len(COLORS)], alpha=0.15)
    val = make_test_set(env.cfg, n=5, base_seed=5000)
    pid_val, _ = evaluate(make_policy("pid", env, "configs/pid.yaml"), env, val)
    ax.axhline(pid_val.rmse.mean(), color="k", ls="--", lw=1, label="PID")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("etkileşim adımı (log)")
    ax.set_ylabel("doğrulama RMSE [°C] (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/methods_learning.png", dpi=130)
    plt.close(fig)
    table = pd.DataFrame(rows).set_index("yöntem")
    table.to_csv("results/eval/methods.csv")
    Path("results/eval/methods.md").write_text(to_markdown(table), encoding="utf-8")
    print(table.to_string())


if __name__ == "__main__":
    main()
