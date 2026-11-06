"""Ödül şekillendirmenin yan etkileri için ölçümler (rubrik: formülasyon).

1. Erken bitiş açığı: 950 °C aşımı bölümü bitirip tek seferlik ceza verseydi, fırını kasten aşırı ısıtan
   sabit bir kural PID'den yüksek ödül alır mıydı? Eğitim gerektirmez, doğrudan ölçülür.
2. Isı deposu: ana ajan B1'i parti gelmeden önce ve diğer zamanlarda hedefin ne kadar üstünde tutuyor?
   (Önce scripts/compare_e3.py --tag e3 çalıştırılmalıdır.)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import load_scenarios
from furnace_rl.evaluate import evaluate, load_traj, make_policy
from furnace_rl.plots import to_markdown

EARLY_END_PENALTY = 10.0


def early_end_return(env: FurnaceEnv, scenario: dict) -> tuple[float, int]:
    """Tam güç kuralı; ilk 950 °C aşımında bölüm biter (erken bitişli tasarım)."""
    env.reset(seed=scenario["seed"], options={"scenario": scenario["schedule"]})
    total, t = 0.0, 0
    while True:
        _, r, _, trunc, info = env.step(np.ones(env.n, dtype=np.float32))
        total += r
        t += 1
        if np.any(info["T_true"] > env.cfg["T_max"]):
            return total - EARLY_END_PENALTY, t
        if trunc:
            return total, t


def heat_reserve(run_tag: str, names: list[str]) -> dict:
    pre, other = [], []
    for name in names:
        tr = load_traj(f"results/eval/{run_tag}/trajs/{name}.npz")
        err = (tr["T_true"] - tr["T_ref"])[:, 0]
        mask = np.zeros(len(err), dtype=bool)
        for ta in tr["arrivals"]:
            mask |= (tr["t"] > ta - 600) & (tr["t"] <= ta)
        load = tr["zone_load"][:, 0] > 0
        pre.append(err[mask].mean() if mask.any() else np.nan)
        other.append(err[~mask & ~load].mean() if (~mask & ~load).any() else np.nan)
    return {"varış öncesi 10 dk B1 sapma [°C]": float(np.nanmean(pre)),
            "yüksüz diğer zaman B1 sapma [°C]": float(np.nanmean(other))}


def main():
    env = FurnaceEnv()
    scenarios = load_scenarios("configs/test_scenarios.json")
    pid, _ = evaluate(make_policy("pid", env, "configs/pid.yaml"), env, scenarios)
    res = [early_end_return(env, s) for s in scenarios]
    rows = [{"politika": "PID, tüm bölüm", "ödül": pid["return"].mean(), "bölüm uzunluğu [adım]": 720},
            {"politika": "Tam güç, erken bitişli tasarım", "ödül": np.mean([r[0] for r in res]),
             "bölüm uzunluğu [adım]": np.mean([r[1] for r in res])}]
    t1 = pd.DataFrame(rows).set_index("politika").round(1)
    print(t1.to_string())
    names = [s["name"] for s in scenarios]
    rows2 = [{"politika": "PID", **heat_reserve("e3_pid", names)}]
    for s in range(3):
        rows2.append({"politika": f"MB-RL seed {s}", **heat_reserve(f"e3_mbrl_seed{s}", names)})
    t2 = pd.DataFrame(rows2).set_index("politika").round(1)
    print(t2.to_string())
    Path("results/eval/reward_analysis.md").write_text(to_markdown(t1) + "\n\n" + to_markdown(t2), encoding="utf-8")


if __name__ == "__main__":
    main()
