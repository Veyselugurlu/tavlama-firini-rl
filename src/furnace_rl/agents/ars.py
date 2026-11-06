"""ARS: Augmented Random Search ile doğrusal politika eğitimi (sinir ağı yok).

Mania, Guy & Recht (2018), "Simple random search of static linear policies is competitive for
reinforcement learning". Modelden bağımsız politika arama yöntemidir:

    a = clip(K z(s), -1, 1)
    Her turda N rastgele yön δ çekilir; K ± νδ politikaları aynı senaryoda koşturulur.
    K ← K + α / (b σ_R) · Σ_{en iyi b yön} (R⁺ − R⁻) δ

Bu, ödülün K'ya göre gradyanının sonlu fark tahminidir. Ajan fizik denklemlerini görmez, yalnızca
etkileşimden elde edilen bölüm ödülünü kullanır.

Örnek:
    python -m furnace_rl.agents.ars --seed 0
    python -m furnace_rl.agents.ars --seed 0 --set lr=0.01 --name e5_lr0.01_seed0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.config import load_env_config, load_yaml, merge_config
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import make_test_set
from furnace_rl.evaluate import evaluate

EVAL_KEYS = ("return", "rmse", "max_drop", "overshoot", "energy_kwh", "smoothness", "unsafe_min")
N_FEATURES = 15


def state_features(obs: np.ndarray, slices: dict) -> np.ndarray:
    """z(s) = [hata, türev, yük, u_prev − u_eff, m, m·τ, m·τ²] (15 boyut, sabit terim yok).

    Eylem güçteki değişim olduğu için doğrusal politika "hız biçimli" PI'ye karşılık gelir: yüksüz
    dengede tüm özellikler sıfırdır, dolayısıyla a = 0 ve kalıcı durum hatası oluşmaz. u_prev − u_eff
    "yoldaki" ısıyı, m·τ terimleri gelecek partinin büyüklüğü ve yakınlığını temsil eder.
    """
    obs = np.atleast_2d(obs)
    preview = obs[:, slices["preview"]]
    m, tau = preview[:, 0:1], preview[:, 2:3]
    return np.hstack([
        obs[:, slices["T_err"]],
        obs[:, slices["dT"]],
        obs[:, slices["load"]],
        obs[:, slices["u_prev"]] - obs[:, slices["u_eff"]],
        m, m * tau, m * tau ** 2,
    ])


class LinearPolicy:
    """a = clip(K z, -1, 1). `predict` imzası değerlendirme kodu içindir."""

    def __init__(self, K: np.ndarray, slices: dict):
        self.K = K
        self.slices = slices

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        z = state_features(np.asarray(obs, float), self.slices)[0]
        return np.clip(self.K @ z, -1.0, 1.0).astype(np.float32), None

    def save(self, path: str | Path) -> None:
        np.savez(path, K=self.K)

    @classmethod
    def load(cls, path: str | Path, slices: dict) -> "LinearPolicy":
        return cls(np.load(path)["K"], slices)


# ------------------------------------------------------------------ paralel koşum
_worker: dict = {}


def _init_worker(env_cfg: dict) -> None:
    _worker["env"] = FurnaceEnv(env_cfg)


def _rollout(args: tuple[np.ndarray, int]) -> float:
    K, seed = args
    env: FurnaceEnv = _worker["env"]
    slices = env.obs_slices
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        z = state_features(obs, slices)[0]
        obs, r, term, trunc, _ = env.step(np.clip(K @ z, -1.0, 1.0).astype(np.float32))
        total += r
        done = term or trunc
    return total


def train(seed: int, cfg: dict, env_cfg: dict, out_dir: Path, workers: int = 4) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"seed": seed, "ars": cfg, "env": env_cfg}, indent=2, ensure_ascii=False), encoding="utf-8")

    rng = np.random.default_rng(seed)
    env = FurnaceEnv(env_cfg)
    slices = env.obs_slices
    val = make_test_set(env_cfg, n=cfg["n_val_scenarios"], base_seed=cfg["val_base_seed"])
    K = np.zeros((env.n, N_FEATURES))
    N, b, nu = cfg["n_directions"], cfg["top_directions"], cfg["nu"]
    best, rows, t0, steps = -np.inf, [], time.time(), 0

    with mp.Pool(workers, initializer=_init_worker, initargs=(env_cfg,)) as pool:
        for it in range(cfg["n_iterations"]):
            delta = rng.normal(size=(N, *K.shape))
            # Eşlenik ± politikalar aynı senaryoyu görür: senaryo farkı gradyan tahminine karışmaz.
            seeds = rng.integers(2 ** 31, size=N * cfg["episodes_per_direction"])
            jobs = []
            for i in range(N):
                for s in seeds[i * cfg["episodes_per_direction"]: (i + 1) * cfg["episodes_per_direction"]]:
                    jobs.append((K + nu * delta[i], int(s)))
                    jobs.append((K - nu * delta[i], int(s)))
            res = np.array(pool.map(_rollout, jobs, chunksize=2)).reshape(N, cfg["episodes_per_direction"], 2).mean(1)
            steps += len(jobs) * env.cfg["episode_steps"]
            r_pos, r_neg = res[:, 0], res[:, 1]
            top = np.argsort(np.maximum(r_pos, r_neg))[-b:]
            sigma_r = np.concatenate([r_pos[top], r_neg[top]]).std() + 1e-8
            K = K + cfg["lr"] / (b * sigma_r) * np.sum((r_pos[top] - r_neg[top])[:, None, None] * delta[top], axis=0)
            lr_step = cfg["lr"]

            if (it + 1) % cfg["eval_every"] == 0 or it == 0:
                policy = LinearPolicy(K, slices)
                df, _ = evaluate(policy, env, val)
                row = {"iteration": it, "step": steps, "wall_min": (time.time() - t0) / 60.0,
                       **{k: float(df[k].mean()) for k in EVAL_KEYS}}
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "evaluations.csv", index=False)
                star = ""
                if row["return"] > best:
                    best = row["return"]
                    policy.save(out_dir / "best_model.npz")
                    star = "  *"
                policy.save(out_dir / "final_model.npz")
                print(f"[{it:>4}] {steps:>8} adım {row['wall_min']:5.1f} dk  ödül {row['return']:9.1f}  RMSE "
                      f"{row['rmse']:6.1f}  düşüş {row['max_drop']:6.1f}  aşma {row['overshoot']:5.1f}  "
                      f"kWh {row['energy_kwh']:6.1f}  |K| {np.abs(K).max():.2f}{star}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/ars.yaml")
    ap.add_argument("--env-config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--name", help="çıktı klasörü adı (varsayılan ars_seed{seed})")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        key, val = kv.split("=", 1)
        overrides[key] = json.loads(val)
    cfg = merge_config(load_yaml(args.config), overrides)
    env_cfg = load_env_config(args.env_config)
    train(args.seed, cfg, env_cfg, Path("results/models") / (args.name or f"ars_seed{args.seed}"), args.workers)


if __name__ == "__main__":
    main()
