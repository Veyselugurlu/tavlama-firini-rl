"""E1 – PID kazançlarının ızgara araması (docs/04_deney_ve_degerlendirme.md §2).

Ayar senaryoları test setinden farklı seed'lerle (0..N−1) üretilir; amaç RL ile aynı ödülün
bölüm ortalaması. Tüm bölgeler aynı kazançları kullanır.

Ödül yüzeyi kazançlara karşı oldukça düz olduğu için yalnızca ödüle bakmak, yüksüz hâlde bile
salınan (sahada kabul edilmeyecek) ayarları seçebiliyor. Bu yüzden önce kararlılık şartı uygulanır:
yüksüz, hedefin 30 °C altından başlayan adım yanıtında son 30 dk'daki tepe-tepe salınım < 2 °C.
Şartı sağlayanlar arasından en yüksek ödüllü olan configs/pid.yaml'a yazılır.
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from furnace_rl.baselines.pid import PIDPolicy
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.evaluate import run_episode
from furnace_rl.metrics import episode_metrics

KP = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]  # güç oranı / °C
TI = [60.0, 150.0, 300.0, 600.0, 1200.0]   # s
TD = [0.0, 30.0, 60.0]                      # s


OSC_LIMIT = 2.0   # °C


def step_response(kp, ti, td) -> dict:
    env = FurnaceEnv({"noise_std": 0.0, "T_amb_jitter": 0.0, "episode_steps": 540})
    tr = run_episode(env, PIDPolicy(env, kp, ti, td), [], seed=0, options={"T_init": env.T_ref - 30.0})
    tail = tr["T_true"][-180:]
    return {"oscillation": float(np.max(tail.max(0) - tail.min(0))),
            "step_overshoot": float(max(0.0, (tr["T_true"] - tr["T_ref"]).max()))}


def score(args):
    kp, ti, td, seeds = args
    env = FurnaceEnv()
    ms = [episode_metrics(run_episode(env, PIDPolicy(env, kp, ti, td), seed=s), env.dt) for s in seeds]
    return {"kp": kp, "ti": ti, "td": td, **step_response(kp, ti, td),
            **{k: float(np.mean([m[k] for m in ms]))
               for k in ("return", "rmse", "max_drop", "overshoot", "energy_kwh", "smoothness")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    seeds = list(range(args.episodes))
    grid = [(kp, ti, td, seeds) for kp, ti, td in itertools.product(KP, TI, TD)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(score, grid))
    df = pd.DataFrame(rows).sort_values("return", ascending=False)
    Path("results/eval").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/eval/e1_pid_grid.csv", index=False)
    stable = df[df.oscillation < OSC_LIMIT]
    print(f"kararlı ayar: {len(stable)}/{len(df)}")
    print("en iyi 5 (tümü):")
    print(df.head(5).round(3).to_string(index=False))
    print("en iyi 10 (kararlı):")
    print(stable.head(10).round(3).to_string(index=False))

    best = stable.iloc[0]
    gains = {"kp": float(best.kp), "ti": float(best.ti), "td": float(best.td)}
    with open("configs/pid.yaml", "w", encoding="utf-8") as f:
        f.write("# scripts/tune_pid.py ile ızgara aramasından (E1) seçildi –\n"
                "# ölçüt: kararlı ayarlar (salınım < 2 °C) arasında en yüksek ortalama bölüm ödülü\n")
        yaml.safe_dump(gains, f)
    print("en iyi:", gains)


if __name__ == "__main__":
    main()
