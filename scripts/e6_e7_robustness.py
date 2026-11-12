"""E6 – sensör gürültüsü testi ve E7 – dağılım dışı senaryolar (yeniden eğitim yok).

E6: PID ve RL ajanı (her seed'in en iyi modeli) σ_noise ∈ {0, 1, 3, 5} °C ile 20 test senaryosunda.
E7: aynı politikalar configs/special_scenarios.json'daki elle tanımlı zor senaryolarda.

Örnek:
    python scripts/e6_e7_robustness.py --runs results/models/q_table_seed0 results/models/q_table_seed1 results/models/q_table_seed2
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from furnace_rl.envs.furnace_env import FurnaceEnv  # noqa: E402
from furnace_rl.envs.scenarios import load_scenarios  # noqa: E402
from furnace_rl.evaluate import evaluate, make_policy  # noqa: E402
from furnace_rl.plots import COLORS, to_markdown  # noqa: E402

NOISE_LEVELS = [0.0, 1.0, 3.0, 5.0]
KEYS = ["return", "rmse", "max_drop", "overshoot", "energy_kwh", "smoothness"]


def policies(env, runs):
    yield "PID", make_policy("pid", env, "configs/pid.yaml")
    for r in runs:
        yield Path(r).name, make_policy("rl", env, model=r)


def e6(runs, scenarios, out: Path) -> pd.DataFrame:
    rows = []
    for sigma in NOISE_LEVELS:
        env = FurnaceEnv({"noise_std": sigma})
        for name, pol in policies(env, runs):
            df, _ = evaluate(pol, env, scenarios)
            rows.append({"sigma": sigma, "policy": name, **df[KEYS].mean().to_dict()})
            print(f"σ={sigma}  {name:14s} RMSE {rows[-1]['rmse']:5.1f}  ödül {rows[-1]['return']:8.1f}  "
                  f"titreşim {rows[-1]['smoothness']:.3f}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "e6_noise_raw.csv", index=False)

    raw["group"] = raw["policy"].where(raw["policy"] == "PID", "Q-tablosu")
    agg = raw.groupby(["sigma", "group"])[KEYS].agg(["mean", "std"])
    agg.to_csv(out / "e6_noise.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, title in zip(axes, ["rmse", "return", "smoothness"],
                              ["RMSE [°C]", "Bölüm ödülü", "Eylem titreşimi"]):
        for j, g in enumerate(["PID", "Q-tablosu"]):
            mu = agg.xs(g, level="group")[(key, "mean")]
            sd = agg.xs(g, level="group")[(key, "std")].fillna(0)
            ax.errorbar(mu.index, mu.values, yerr=sd.values, marker="o", capsize=3, color=COLORS[j], label=g)
        ax.set_xlabel("termokupl gürültüsü σ [°C]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig("results/figures/e6_noise.png", dpi=130)
    plt.close(fig)
    return agg


def e7(runs, out: Path) -> pd.DataFrame:
    env = FurnaceEnv()
    special = load_scenarios("configs/special_scenarios.json")
    frames = []
    for name, pol in policies(env, runs):
        df, _ = evaluate(pol, env, special)
        df["policy"] = name
        frames.append(df)
    raw = pd.concat(frames)
    raw.to_csv(out / "e7_special_raw.csv")
    raw["group"] = raw["policy"].where(raw["policy"] == "PID", "Q-tablosu")
    table = raw.reset_index().groupby(["scenario", "group"])[["return", "rmse", "max_drop", "overshoot",
                                                              "energy_kwh"]].mean().round(1)
    table.to_csv(out / "e7_special.csv")
    (out / "e7_special.md").write_text(to_markdown(table.reset_index().set_index("scenario")), encoding="utf-8")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scenarios", default="configs/test_scenarios.json")
    args = ap.parse_args()
    out = Path("results/eval")
    out.mkdir(parents=True, exist_ok=True)

    agg = e6(args.runs, load_scenarios(args.scenarios), out)
    print("\nE6 – gürültü (ortalama):")
    print(agg.xs("mean", axis=1, level=1)[["rmse", "return", "max_drop", "smoothness"]].round(2).to_string())
    print("\nE7 – dağılım dışı senaryolar:")
    print(e7(args.runs, out).to_string())


if __name__ == "__main__":
    main()
