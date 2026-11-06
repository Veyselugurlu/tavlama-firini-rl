"""Model tabanlı RL (MPC-RL): doğrusal öğrenilmiş model + CEM planlama (sinir ağı yok).

Döngü (PETS'in veri toplama-model-planlama döngüsüyle aynı, model olarak sinir ağı yerine ridge regresyon):
    1. Rastgele keşifle veri topla.
    2. Geçiş modelini tüm veriden en küçük karelerle öğren.
    3. Modelle CEM üzerinden planlayan ajanı fırında koştur, yeni veriyi ekle, 2'ye dön.

Öğrenilen model (ısıl kapasite, kayıp, iletim, ısı transferi ve aktüatör zaman sabiti ajana verilmez):
    e_{t+1} = A e_t + B u_eff,t + D l_t + c + b_t      (hata, 3 bölge)
    u_eff,{t+1} = α u_eff,t + β u_{t−d}                (ısıtıcı gecikmesi, bölge başına)
l_t: bölge yükü özellikleri. b_t: anlık model hatasının kayan ortalaması (bozucu kestirimi).

Ödül fonksiyonu tasarımcı tarafından tanımlandığı için planlayıcı onu bilir (PETS'te de böyledir). Ölü zaman
adım sayısı, eylem sınırı ve partinin B1'den girdiği de tasarım bilgisi olarak verilir.
Gelecek parti yalnızca önizleme bilgisinden bilinir (varış anında B1 yüküne eklenir).

Örnek:
    python -m furnace_rl.agents.mbrl --seed 0
    python -m furnace_rl.agents.mbrl --seed 0 --set horizon=30 --name e5_h30_seed0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.config import load_env_config, load_yaml, merge_config
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import make_test_set
from furnace_rl.evaluate import evaluate

EVAL_KEYS = ("return", "rmse", "max_drop", "overshoot", "energy_kwh", "smoothness", "unsafe_min")


class LoadTracker:
    """Yük özellikleri: l = [yük, taze yük]. Parti ısındıkça daha az ısı çeker; iç sıcaklığı ölçülemediği
    için bölgeye yeni giren kütle üstel unutmalı bir süzgeçle izlenir."""

    def __init__(self, n: int, rho: float):
        self.n, self.rho = n, rho
        self.reset()

    def reset(self):
        self.prev = np.zeros(self.n)
        self.fresh = np.zeros(self.n)

    def update(self, load: np.ndarray) -> np.ndarray:
        self.fresh = self.rho * self.fresh + np.maximum(load - self.prev, 0.0)
        self.prev = load.copy()
        return np.concatenate([load, self.fresh])


class LinearModel:
    """Ridge regresyonla öğrenilen doğrusal geçiş modeli."""

    def __init__(self, n: int, lam: float):
        self.n, self.lam = n, lam
        self.W_e = None      # e' − e = W_e [e, u_eff, l, 1]
        self.alpha = None
        self.beta = None

    def fit(self, data: dict) -> None:
        X = np.hstack([data["e"], data["u_eff"], data["l"], np.ones((len(data["e"]), 1))])
        Y = data["e_next"] - data["e"]
        self.W_e = np.linalg.solve(X.T @ X + self.lam * np.eye(X.shape[1]), X.T @ Y)
        self.alpha, self.beta = np.zeros(self.n), np.zeros(self.n)
        for i in range(self.n):
            X_u = np.column_stack([data["u_eff"][:, i], data["u_del"][:, i]])
            sol = np.linalg.solve(X_u.T @ X_u + 1e-6 * np.eye(2), X_u.T @ data["u_eff_next"][:, i])
            self.alpha[i], self.beta[i] = sol

    def delta_e(self, e: np.ndarray, u_eff: np.ndarray, l: np.ndarray) -> np.ndarray:
        """Toplu tahmin: girişler (P, n), (P, n), (P veya 1, 2n)."""
        P = e.shape[0]
        X = np.hstack([e, u_eff, np.broadcast_to(l, (P, l.shape[-1])), np.ones((P, 1))])
        return X @ self.W_e

    def to_dict(self) -> dict:
        return {"W_e": self.W_e, "alpha": self.alpha, "beta": self.beta}

    @classmethod
    def load(cls, path: str | Path, n: int, lam: float = 0.0) -> "LinearModel":
        z = np.load(path)
        m = cls(n, lam)
        m.W_e, m.alpha, m.beta = z["W_e"], z["alpha"], z["beta"]
        return m


class MPCPolicy:
    """Öğrenilmiş modelle CEM planlama. Her `replan` adımda plan yenilenir (kayan ufuk)."""

    def __init__(self, model: LinearModel, env: FurnaceEnv, cfg: dict, seed: int = 0):
        self.model, self.env, self.cfg = model, env, cfg
        self.sl = env.obs_slices
        self.n, self.d = env.n, int(env.cfg["dead_steps"])
        self.du = env.cfg["du_max"]
        self.H = cfg["horizon"]
        self.seg = cfg["segment"]
        self.n_seg = self.H // self.seg
        self.err_scale = env.cfg["obs"]["T_err_scale"]
        self.mass_scale = env.cfg["obs"]["mass_scale"]
        self.preview_h = env.cfg["batch"]["preview_min"] * 60.0 / env.dt
        rc = env.cfg["reward"]
        self.w_t, self.w_e, self.w_s = rc["w_track"], rc["w_energy"], rc["w_smooth"]
        self.dT_scale = rc["dT_scale"]
        self.safety = rc["safety_penalty"]
        self.margin = (env.cfg["T_max"] - env.T_ref) / self.err_scale
        self.power_share = env.P_max * env.dt / 3.6e6 / env.E_max
        self.gammas = cfg["gamma"] ** np.arange(self.H)
        self.tracker = LoadTracker(self.n, cfg["fresh_rho"])
        self.seed = seed
        self.sigma_extra = 0.0
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)   # aynı senaryo aynı planı üretsin
        self.t = 0
        self.tracker.reset()
        self.mu = np.zeros((self.n_seg, self.n))
        self.bias = np.zeros(self.n)
        self.prev = None
        self.a = np.zeros(self.n)

    def _observe(self, obs: np.ndarray) -> dict:
        s = self.sl
        return {"e": obs[s["T_err"]], "u_eff": obs[s["u_eff"]], "u_prev": obs[s["u_prev"]],
                "pending": obs[s["pending"]].reshape(self.d, self.n), "l": self.tracker.update(obs[s["load"]]),
                "preview": obs[s["preview"]]}

    def _load_forecast(self, o: dict) -> np.ndarray:
        """Ufuk boyunca yük özellikleri (H, 2n). Mevcut yük sabit, önizlemedeki parti varışta B1'e eklenir."""
        L = np.tile(o["l"], (self.H, 1))
        n = self.n
        L[:, n:] = o["l"][n:] * self.cfg["fresh_rho"] ** np.arange(self.H)[:, None]
        m, _, tau = o["preview"]
        if m > 0:
            k = int(round(tau * self.preview_h))
            if k < self.H:
                L[k:, 0] += m
                L[k:, n] += m * self.cfg["fresh_rho"] ** np.arange(self.H - k)
        return L

    def _rollout_cost(self, A: np.ndarray, o: dict, L: np.ndarray) -> np.ndarray:
        """A: (P, n_seg, n) normalize eylem dizileri. Dönüş: (P,) negatif olmayan indirimli maliyet."""
        P = A.shape[0]
        m = self.model
        e = np.tile(o["e"], (P, 1))
        u_eff = np.tile(o["u_eff"], (P, 1))
        u = np.tile(o["u_prev"], (P, 1))
        queue = [np.tile(o["pending"][j], (P, 1)) for j in range(self.d)]
        cost = np.zeros(P)
        for t in range(self.H):
            u_new = np.clip(u + self.du * A[:, t // self.seg], 0.0, 1.0)
            cost += self.gammas[t] * self.w_s * np.abs(u_new - u).mean(axis=1)
            u = u_new
            if self.d:
                queue.append(u)
                u_del = queue.pop(0)
            else:
                u_del = u
            de = m.delta_e(e, u_eff, L[t]) + self.bias
            u_eff = m.alpha * u_eff + m.beta * u_del
            e = e + de
            err = e * self.err_scale / self.dT_scale
            step = self.w_t * (err ** 2).mean(axis=1) + self.w_e * (u_eff @ self.power_share)
            step = step + self.safety * np.any(e > self.margin, axis=1)
            cost += self.gammas[t] * step
        return cost

    def _plan(self, o: dict) -> np.ndarray:
        c = self.cfg
        L = self._load_forecast(o)
        mu = self.mu.copy()
        sigma = np.full_like(mu, c["cem_sigma"])
        n_elite = max(2, int(c["cem_elite"] * c["cem_pop"]))
        for _ in range(c["cem_iters"]):
            A = np.clip(mu + sigma * self.rng.normal(size=(c["cem_pop"], *mu.shape)), -1.0, 1.0)
            A[0] = np.clip(mu, -1.0, 1.0)
            elite = A[np.argsort(self._rollout_cost(A, o, L))[:n_elite]]
            mu = 0.2 * mu + 0.8 * elite.mean(axis=0)
            sigma = 0.2 * sigma + 0.8 * elite.std(axis=0)
        self.mu = np.vstack([mu[1:], mu[-1:]]) if c["replan"] >= c["segment"] else mu
        return mu[0]

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        o = self._observe(np.asarray(obs, float))
        # Bozucu kestirimi: son adımın model hatası (ölçüm gürültüsü süzülerek)
        if self.prev is not None:
            pred = self.prev["e"] + self.model.delta_e(self.prev["e"][None], self.prev["u_eff"][None], self.prev["l"])[0]
            r = self.cfg["bias_filter"]
            self.bias = (1 - r) * self.bias + r * (o["e"] - pred)
        if self.t % self.cfg["replan"] == 0:
            self.a = self._plan(o)
        self.prev = o
        self.t += 1
        return np.clip(self.a + self.sigma_extra * self.rng.normal(size=self.n), -1, 1).astype(np.float32), None


def empty_data() -> dict:
    return {k: [] for k in ("e", "u_eff", "l", "e_next", "u_del", "u_eff_next")}


def collect(env: FurnaceEnv, cfg: dict, rng: np.random.Generator, policy: MPCPolicy | None, data: dict) -> None:
    """Bir bölüm toplar. `policy` yoksa başlangıç gücü etrafında rastgele keşif yapılır."""
    sl = env.obs_slices
    obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
    tracker = LoadTracker(env.n, cfg["fresh_rho"])
    u0 = obs[sl["u_prev"]].copy()
    noise = np.zeros(env.n)
    if policy is not None:
        policy.reset()
    l = tracker.update(obs[sl["load"]])
    step, done = 0, False
    while not done:
        if policy is not None:
            a, _ = policy.predict(obs)
            a = np.clip(a + cfg["explore_sigma_mpc"] * rng.normal(size=env.n), -1, 1).astype(np.float32)
        else:
            if step % cfg["noise_hold"] == 0:
                noise = rng.normal(0.0, cfg["explore_sigma"], env.n)
            a = np.clip(-cfg["rho"] * (obs[sl["u_prev"]] - u0) / env.cfg["du_max"] + noise, -1, 1).astype(np.float32)
        u_del = obs[sl["pending"]].reshape(int(env.cfg["dead_steps"]), env.n)[0]
        obs2, _, term, trunc, _ = env.step(a)
        l2 = tracker.update(obs2[sl["load"]])
        data["e"].append(obs[sl["T_err"]]); data["u_eff"].append(obs[sl["u_eff"]]); data["l"].append(l)
        data["e_next"].append(obs2[sl["T_err"]]); data["u_del"].append(u_del); data["u_eff_next"].append(obs2[sl["u_eff"]])
        obs, l = obs2, l2
        step += 1
        done = term or trunc


def stacked(data: dict) -> dict:
    return {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}


def train(seed: int, cfg: dict, env_cfg: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"seed": seed, "mbrl": cfg, "env": env_cfg}, indent=2, ensure_ascii=False), encoding="utf-8")
    rng = np.random.default_rng(seed)
    env = FurnaceEnv(env_cfg)
    val_env = FurnaceEnv(env_cfg)
    val = make_test_set(env_cfg, n=cfg["n_val_scenarios"], base_seed=cfg["val_base_seed"])
    data = empty_data()
    model = LinearModel(env.n, cfg["lam"])
    best, rows, t0 = -np.inf, [], time.time()

    for it in range(cfg["n_iterations"]):
        n_new = cfg["init_episodes"] if it == 0 else cfg["episodes_per_iter"]
        policy = None if it == 0 else MPCPolicy(model, env, cfg, seed=int(rng.integers(2 ** 31)))
        for _ in range(n_new):
            collect(env, cfg, rng, policy, data)
        model.fit(stacked(data))

        df, _ = evaluate(MPCPolicy(model, val_env, cfg, seed=seed), val_env, val)
        steps = len(data["e"])
        row = {"iteration": it, "step": steps, "wall_min": (time.time() - t0) / 60.0,
               **{k: float(df[k].mean()) for k in EVAL_KEYS}}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "evaluations.csv", index=False)
        star = ""
        np.savez(out_dir / "final_model.npz", **model.to_dict())
        if row["return"] > best:
            best = row["return"]
            np.savez(out_dir / "best_model.npz", **model.to_dict())
            star = "  *"
        print(f"[{it:>3}] {steps:>7} adım {row['wall_min']:5.1f} dk  ödül {row['return']:9.1f}  RMSE {row['rmse']:6.1f}  "
              f"düşüş {row['max_drop']:6.1f}  aşma {row['overshoot']:5.1f}  kWh {row['energy_kwh']:6.1f}  "
              f"β {np.round(model.beta, 4)}{star}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/mbrl.yaml")
    ap.add_argument("--env-config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--name")
    args = ap.parse_args()
    overrides = {}
    for kv in args.set:
        key, val = kv.split("=", 1)
        overrides[key] = json.loads(val)
    cfg = merge_config(load_yaml(args.config), overrides)
    env_cfg = load_env_config(args.env_config)
    train(args.seed, cfg, env_cfg, Path("results/models") / (args.name or f"mbrl_seed{args.seed}"))


if __name__ == "__main__":
    main()
