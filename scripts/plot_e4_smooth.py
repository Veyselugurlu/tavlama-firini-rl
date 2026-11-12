"""E4 – eylem titreşimi cezasının (w₃) etkisi: w₃ = 0 ile eğitilmiş ajan, ana ajana (w₃ = 0.05) karşı (3 seed).

Titreşim iki ölçüyle verilir: ortalama |Δu| (smoothness) ve güç komutunun yön değiştirme oranı
(ardışık iki adımda Δu'nun işaret değiştirdiği adımların oranı, |Δu| > 0.01 olanlar sayılır).
Bir örnek senaryoda B1 güç komutu zaman serisi olarak çizilir.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from furnace_rl.envs.furnace_env import FurnaceEnv  # noqa: E402
from furnace_rl.envs.scenarios import load_scenarios  # noqa: E402
from furnace_rl.evaluate import evaluate, make_policy  # noqa: E402
from furnace_rl.plots import COLORS, to_markdown  # noqa: E402

RUNS = {"PID": None, "Q-tablosu w₃=0.05 (referans)": "q_table_seed{}", "Q-tablosu w₃=0": "e4q_wsmooth0.0_seed{}"}
EXAMPLE, WINDOW_MIN = "test_05", (20, 60)


def reversals(u: np.ndarray, eps: float = 0.01) -> float:
    du = np.diff(u, axis=0)
    du[np.abs(du) < eps] = 0.0
    flips = (du[1:] * du[:-1]) < 0
    return float(flips.mean())


def main():
    env = FurnaceEnv()
    scenarios = load_scenarios("configs/test_scenarios.json")
    rows, examples = [], {}
    for label, run in RUNS.items():
        pols = ([make_policy("pid", env, "configs/pid.yaml")] if run is None else
                [make_policy("rl", env, model=f"results/models/{run.format(s)}") for s in range(3)])
        res = [evaluate(pol, env, scenarios) for pol in pols]
        df = pd.concat([r[0] for r in res])
        rev = np.mean([reversals(tr["u_cmd"]) for r in res for tr in r[1]])
        rows.append({"politika": label, "RMSE [°C]": df.rmse.mean(), "maks. düşüş [°C]": df.max_drop.mean(),
                     "enerji [kWh]": df.energy_kwh.mean(), "ort. |Δu|": df.smoothness.mean(),
                     "yön değişimi oranı": rev})
        examples[label] = res[0][1][list(res[0][0].index).index(EXAMPLE)]
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).set_index("politika").round(3)
    table.to_csv("results/eval/e4_smooth.csv")
    Path("results/eval/e4_smooth.md").write_text(to_markdown(table), encoding="utf-8")
    print(table.to_string())

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.2 * len(examples)), sharex=True, sharey=True)
    for ax, (j, (label, tr)) in zip(axes, enumerate(examples.items())):
        t = tr["t"] / 60.0
        m = (t >= WINDOW_MIN[0]) & (t <= WINDOW_MIN[1])
        ax.step(t[m], tr["u_cmd"][m, 0], where="post", color=COLORS[j], lw=1.2)
        for ta in tr["arrivals"]:
            if WINDOW_MIN[0] <= ta / 60 <= WINDOW_MIN[1]:
                ax.axvline(ta / 60, color="k", ls=":", lw=1)
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_ylabel("B1 güç komutu")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f"zaman [dk] ({EXAMPLE}; noktalı çizgi: parti varışı)")
    fig.tight_layout()
    fig.savefig("results/figures/e4_smooth.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
