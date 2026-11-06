"""E4 – enerji ağırlığı (w₂) taraması: izleme hatası ile enerji arasındaki ödünleşim (3 seed).

Her w₂ için ajan aynı veri toplama döngüsüyle eğitilir; planlayıcı kendi w₂'siyle çalışır. Fiziksel
metrikler (RMSE, kWh) varsayılan ortamda, 20 test senaryosunda ölçülür. w₂ = 0.2 ana deneydir (mbrl_seed*).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from furnace_rl.envs.furnace_env import FurnaceEnv  # noqa: E402
from furnace_rl.envs.scenarios import load_scenarios  # noqa: E402
from furnace_rl.evaluate import evaluate, make_policy  # noqa: E402
from furnace_rl.plots import COLORS, to_markdown  # noqa: E402

GROUPS = {0.0: "e4_wenergy0.0_seed{}", 0.1: "e4_wenergy0.1_seed{}", 0.2: "mbrl_seed{}",
          0.5: "e4_wenergy0.5_seed{}", 1.0: "e4_wenergy1.0_seed{}"}
KEYS = ["rmse", "max_drop", "overshoot", "energy_kwh", "smoothness"]


def main():
    env = FurnaceEnv()
    scenarios = load_scenarios("configs/test_scenarios.json")
    rows = []
    df, _ = evaluate(make_policy("pid", env, "configs/pid.yaml"), env, scenarios)
    rows.append({"politika": "PID", "w2": None, **df[KEYS].mean().round(2).to_dict()})
    for w, pattern in GROUPS.items():
        means = []
        for s in range(3):
            d, _ = evaluate(make_policy("rl", env, model=f"results/models/{pattern.format(s)}"), env, scenarios)
            means.append(d[KEYS].mean())
        m = pd.DataFrame(means)
        rows.append({"politika": f"MB-RL w₂={w}", "w2": w, **m.mean().round(2).to_dict(),
                     "RMSE std": round(m.rmse.std(), 2), "kWh std": round(m.energy_kwh.std(), 2)})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).set_index("politika")
    table.to_csv("results/eval/e4_energy_weight.csv")
    Path("results/eval/e4_energy_weight.md").write_text(to_markdown(table), encoding="utf-8")
    print(table.to_string())

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rl = table.dropna(subset=["w2"]).sort_values("w2")
    ax.errorbar(rl.energy_kwh, rl.rmse, xerr=rl["kWh std"], yerr=rl["RMSE std"], fmt="o", capsize=3,
                color=COLORS[0], label="MB-RL (3 seed)")
    for _, r in rl.iterrows():
        ax.annotate(f"w₂={r.w2:g}", (r.energy_kwh, r.rmse), textcoords="offset points", xytext=(6, 4), fontsize=9)
    pid = table.loc["PID"]
    ax.plot(pid.energy_kwh, pid.rmse, "s", color=COLORS[1], ms=9, label="PID")
    ax.set_xlabel("enerji [kWh / bölüm] (test seti ort.)")
    ax.set_ylabel("RMSE [°C] (test seti ort.)")
    ax.set_title("İzleme hatası – enerji ödünleşimi", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/e4_pareto.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
