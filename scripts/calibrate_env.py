"""E0 – simülatör kalibrasyonu (docs/03_simulator_tasarimi.md §3 kabul ölçütleri).

1. Yüksüz dengede gereken güç oranı (hedef %30–60)
2. Tam güçte ısınma hızı (hedef 5–20 °C/dk)
3. PID ile 1 tonluk partide bölge 1 düşüşü (hedef 30–80 °C) ve toparlanma (hedef 5–20 dk)
4. Rastgele takvimde PID'in güç doyumu (kapasite yeterli mi?)
"""

from pathlib import Path

import numpy as np

from furnace_rl.baselines.pid import PIDPolicy
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.evaluate import run_episode
from furnace_rl.metrics import episode_metrics, recovery_times
from furnace_rl.plots import plot_episode

PID_CFG = Path("configs/pid.yaml")
ONE_TON = [{"t_arrival": 600.0, "mass": 1000.0, "speed": 0.2}]


class Fixed:
    def __init__(self, u):
        self.a = (2.0 * np.asarray(u, float) - 1.0).astype(np.float32)

    def predict(self, obs, deterministic=True):
        return self.a, None


def make_pid(env):
    return PIDPolicy.from_yaml(env, PID_CFG) if PID_CFG.exists() else PIDPolicy(env)


def main():
    env = FurnaceEnv({"noise_std": 0.0, "T_init_jitter": 0.0, "T_amb_jitter": 0.0, "action_mode": "absolute"})
    env.reset(seed=0, options={"scenario": []})
    u_ss = env.steady_state_power()
    print("1) yüksüz denge gücü [%]:", np.round(100 * u_ss, 1))

    hot = run_episode(FurnaceEnv({"noise_std": 0.0, "T_init_jitter": 0.0, "T_amb_jitter": 0.0,
                                  "action_mode": "absolute", "episode_steps": 90}), Fixed(np.ones(3)), [], seed=0)
    rate = (hot["T_true"][-1] - hot["T_true"][30]) / ((hot["t"][-1] - hot["t"][30]) / 60.0)
    print("2) tam güçte ısınma hızı [°C/dk]:", np.round(rate, 1))

    pid_src = PID_CFG if PID_CFG.exists() else "varsayılan kazançlar"
    open_loop = run_episode(env, Fixed(u_ss), ONE_TON, seed=0)
    pid = run_episode(env, make_pid(env), ONE_TON, seed=0)
    for label, tr in [("sabit güç", open_loop), (f"PID ({pid_src})", pid)]:
        err = tr["T_true"] - tr["T_ref"]
        rec = recovery_times(err[:, 0], tr["t"], tr["arrivals"])
        print(f"3) 1 ton – {label}: düşüş B1/B2/B3 = {np.round(-err.min(0), 1)} °C, "
              f"B1 toparlanma = {rec[0]:.1f} dk, enerji = {tr['energy_kwh'].sum():.1f} kWh")
    plot_episode({"sabit güç": open_loop, "PID": pid}, "results/figures/e0_calibration_1ton.png",
                 "E0 – 1 tonluk parti, sabit güç ve PID")

    env_r = FurnaceEnv()
    sat, rets = [], []
    for s in range(5):
        tr = run_episode(env_r, make_pid(env_r), seed=s)
        sat.append(np.mean(tr["u_cmd"] >= 0.999, axis=0))
        rets.append(episode_metrics(tr, env_r.dt))
    print("4) rastgele takvim, PID: doyumda geçen süre [%] B1/B2/B3 =",
          np.round(100 * np.mean(sat, 0), 1))
    print("   RMSE [°C] =", np.round([m["rmse"] for m in rets], 1),
          " maks. düşüş [°C] =", np.round([m["max_drop"] for m in rets], 1))


if __name__ == "__main__":
    main()
