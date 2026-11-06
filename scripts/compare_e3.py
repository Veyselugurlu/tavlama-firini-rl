"""E3 – PID ve bir RL yönteminin aynı 20 test senaryosunda kıyası.

Her seed'in en iyi modeli test setinde koşturulur (results/eval/<run>), sonra:
- results/eval/comparison.csv + .md : metrik tablosu (PID: senaryolar arası, RL: seed'ler arası ort. ± std)
- results/figures/e3_bars.png       : RMSE, maks. düşüş, enerji çubuk grafiği
- results/figures/e3_<senaryo>.png  : örnek senaryolarda PID ve RL üst üste
- "önceden ısıtma" göstergesi: partinin varışından önceki 3 dk'da B1 gücünün, 7–10 dk önceki seviyeye göre artışı

Örnek:
    python scripts/compare_e3.py --runs results/models/mbrl_seed0 results/models/mbrl_seed1 results/models/mbrl_seed2 --label "MB-RL"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.config import load_env_config
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import load_scenarios
from furnace_rl.evaluate import evaluate, load_traj, make_policy, save_trajs
from furnace_rl.plots import plot_bars, plot_episode, to_markdown

METRICS = ["return", "rmse", "max_drop", "overshoot", "recovery_min", "energy_kwh", "smoothness", "unsafe_min",
           "preheat"]
NAMES = {"return": "Bölüm ödülü", "rmse": "RMSE [°C]", "max_drop": "Maks. düşüş [°C]", "overshoot": "Aşma [°C]",
         "recovery_min": "B1 toparlanma [dk]", "energy_kwh": "Enerji [kWh]", "smoothness": "Eylem titreşimi",
         "unsafe_min": "950 °C üstünde süre [dk]",
         "preheat": "Varış öncesi B1 güç artışı"}


def preheat(traj: dict, dt: float) -> float:
    """Varıştan önceki 3 dk'daki ortalama B1 güç komutu − 7–10 dk önceki ortalama."""
    u, t = traj["u_cmd"][:, 0], traj["t"]
    vals = []
    for ta in traj["arrivals"]:
        late = (t > ta - 180) & (t <= ta)
        early = (t > ta - 600) & (t <= ta - 420)
        if late.any() and early.any():
            vals.append(u[late].mean() - u[early].mean())
    return float(np.mean(vals)) if vals else 0.0


def run_eval(kind: str, env: FurnaceEnv, scenarios, out: Path, model: str | None = None) -> pd.DataFrame:
    policy = make_policy(kind, env, "configs/pid.yaml", model)
    df, trajs = evaluate(policy, env, scenarios)
    df["preheat"] = [preheat(tr, env.dt) for tr in trajs]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv")
    save_trajs(trajs, list(df.index), out)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--label", default="MB-RL")
    ap.add_argument("--scenarios", default="configs/test_scenarios.json")
    ap.add_argument("--tag", default="e3", help="çıktı dosyası öneki")
    ap.add_argument("--examples", nargs="*", default=["test_05", "test_12"])
    ap.add_argument("--env-config", help="RL koşularının eğitildiği ortam ayarı (ör. önizlemesiz); PID hep varsayılanda")
    args = ap.parse_args()

    env = FurnaceEnv()
    rl_env = FurnaceEnv(load_env_config(args.env_config))
    scenarios = load_scenarios(args.scenarios)
    pid = run_eval("pid", env, scenarios, Path(f"results/eval/{args.tag}_pid"))
    rl = {Path(r).name: run_eval("rl", rl_env, scenarios, Path(f"results/eval/{args.tag}_{Path(r).name}"), r)
          for r in args.runs}

    seed_means = pd.DataFrame({name: df[METRICS].mean() for name, df in rl.items()}).T
    table = pd.DataFrame({
        "PID": [f"{pid[m].mean():.2f} ± {pid[m].std():.2f}" for m in METRICS],
        f"{args.label} (seed ort.)": [f"{seed_means[m].mean():.2f} ± {seed_means[m].std():.2f}" for m in METRICS],
        "Fark [%]": [f"{100 * (seed_means[m].mean() - pid[m].mean()) / abs(pid[m].mean()):+.1f}"
                     if abs(pid[m].mean()) > 1e-9 else "–" for m in METRICS],
    }, index=[NAMES[m] for m in METRICS])
    for name in rl:
        table[name] = [f"{seed_means.loc[name, m]:.2f}" for m in METRICS]
    table.to_csv(f"results/eval/{args.tag}_comparison.csv")
    Path(f"results/eval/{args.tag}_comparison.md").write_text(to_markdown(table), encoding="utf-8")
    print(table.to_string())

    # Seed'lerin senaryo bazında ortalaması ile PID'in senaryo bazında kazanç sayısı
    rl_scn = sum(df["return"] for df in rl.values()) / len(rl)
    print(f"\n{args.label} yönteminin PID'den yüksek ödül aldığı senaryo: {(rl_scn > pid['return']).sum()}/{len(pid)}")

    plot_bars({"PID": pid, args.label: pd.concat(rl.values())}, ["rmse", "max_drop", "energy_kwh"],
              [NAMES[m] for m in ["rmse", "max_drop", "energy_kwh"]], f"results/figures/{args.tag}_bars.png")

    best_run = seed_means["return"].idxmax()
    for sc in args.examples:
        plot_episode({"PID": load_traj(f"results/eval/{args.tag}_pid/trajs/{sc}.npz"),
                      f"{args.label} ({best_run})": load_traj(f"results/eval/{args.tag}_{best_run}/trajs/{sc}.npz")},
                     f"results/figures/{args.tag}_{sc}.png", f"PID ve {args.label} – {sc}")


if __name__ == "__main__":
    main()
