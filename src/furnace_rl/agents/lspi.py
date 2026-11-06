"""LSPI: karesel özelliklerle en küçük kareler politika iterasyonu (sinir ağı yok).

Modelden bağımsız, doğrusal fonksiyon yaklaşımlı klasik RL yöntemi (Lagoudakis & Parr, 2003):
    Q(s, a) = θᵀ φ(s, a),   φ = x = [z(s), a] vektörünün karesel terimleri

- z(s): gözlemden seçilen özellikler (docs/02 §6, `state_features`)
- Politika değerlendirme: LSTD-Q, θ = (Σ φ (φ − γ φ')ᵀ + λI)⁻¹ Σ φ r
- Politika iyileştirme: Q eylemde karesel olduğu için argmax kapalı biçimde, a = K z (sonra [-1, 1] kırpılır)

Simülatörün denklemleri ajana verilmez; ajan yalnızca (s, a, r, s') örneklerinden öğrenir.

Örnek:
    python -m furnace_rl.agents.lspi --seed 0
    python -m furnace_rl.agents.lspi --seed 0 --set lam=1e-2 explore_sigma=0.2 --name e5_lam1e-2_seed0
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


def state_features(obs: np.ndarray, slices: dict) -> np.ndarray:
    """z(s) = [1, hata, türev, yük, fiili güç, bekleyen komutlar, m, m·τ, m·τ²]

    T_ref sabit olduğu, u_prev de bekleyen komutların sonuncusuyla aynı olduğu için alınmaz.
    Önizleme terimleri kütleyi varışa kalan süreyle çarpar: "ne kadar ısıtmalı" sorusunun cevabı
    partinin hem büyüklüğüne hem yakınlığına bağlıdır, Q ise z'de karesel olduğundan bu çarpım
    tek başına oluşmaz. Önizleme yokken m = 0 olduğu için bu terimler sıfırdır.
    """
    obs = np.atleast_2d(obs)
    preview = obs[:, slices["preview"]]
    m, tau = preview[:, 0:1], preview[:, 2:3]
    return np.hstack([
        np.ones((obs.shape[0], 1)),
        obs[:, slices["T_err"]],
        obs[:, slices["dT"]],
        obs[:, slices["load"]],
        obs[:, slices["u_eff"]],
        obs[:, slices["pending"]],
        m, m * tau, m * tau ** 2,
    ])


class LinearPolicy:
    """a = clip(K z, -1, 1). Değerlendirme kodu `predict` imzasını kullanır."""

    def __init__(self, K: np.ndarray, slices: dict):
        self.K = K
        self.slices = slices

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(state_features(obs, self.slices) @ self.K.T, -1.0, 1.0)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.act(np.asarray(obs, float))[0].astype(np.float32), None

    def save(self, path: str | Path) -> None:
        np.savez(path, K=self.K)

    @classmethod
    def load(cls, path: str | Path, slices: dict) -> "LinearPolicy":
        return cls(np.load(path)["K"], slices)


class QuadraticQ:
    """Q(z, a) = xᵀ M x, x = [z, a]. θ, M'nin üst üçgenidir."""

    def __init__(self, n_z: int, n_a: int):
        self.n_z, self.n_a = n_z, n_a
        self.n_x = n_z + n_a
        self.iu = np.triu_indices(self.n_x)
        self.n_features = len(self.iu[0])

    def features(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        x = np.hstack([z, a])
        return x[:, self.iu[0]] * x[:, self.iu[1]]

    def greedy_gain(self, theta: np.ndarray, min_curv: float) -> np.ndarray:
        """argmax_a Q = −M_aa⁻¹ M_az z. M_aa negatif tanımlı değilse eğrilik tabanı uygulanır."""
        M = np.zeros((self.n_x, self.n_x))
        M[self.iu] = theta
        M = (M + M.T) / 2.0
        M_aa = M[self.n_z:, self.n_z:]
        M_az = M[self.n_z:, :self.n_z]
        w, V = np.linalg.eigh(M_aa)
        w = np.minimum(w, -min_curv)
        return -(V @ np.diag(1.0 / w) @ V.T) @ M_az


def lstdq_solve(q: QuadraticQ, Phi: np.ndarray, r: np.ndarray, Z_next: np.ndarray, K: np.ndarray,
                gamma: float, lam: float, chunk: int = 50_000) -> np.ndarray:
    """LSTD-Q: verilen politika K için θ. Φ önceden hesaplanır, yalnızca φ' politikaya bağlıdır."""
    N = Phi.shape[0]
    A = np.zeros((q.n_features, q.n_features))
    for i in range(0, N, chunk):
        P = Phi[i:i + chunk]
        a_next = np.clip(Z_next[i:i + chunk] @ K.T, -1.0, 1.0)
        A += P.T @ (P - gamma * q.features(Z_next[i:i + chunk], a_next))
    A = A / N + lam * np.eye(q.n_features)
    return np.linalg.solve(A, Phi.T @ r / N)


def collect(env: FurnaceEnv, K: np.ndarray | None, slices: dict, sigma: float, hold: int, rho: float,
            rng: np.random.Generator) -> dict:
    """Bir bölüm toplar. K yoksa politika sıfırdır (güç başlangıç değerinde kalır), üstüne gürültü eklenir.
    Gürültü `hold` adım sabit tutulur: ısıl sistem yavaş olduğu için beyaz gürültü dinamiği uyarmaz.
    `rho`: gücün başlangıç seviyesine geri çekilme oranı, rastgele yürüyüşün fırını sınırlardan
    uzaklaştırmasını önler (yalnızca K yokken)."""
    obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
    n = env.n
    zs, acts, rewards, z_nexts = [], [], [], []
    noise = np.zeros(n)
    u0 = obs[slices["u_prev"]].copy()
    z = state_features(obs, slices)[0]
    done = False
    step = 0
    while not done:
        if step % hold == 0:
            noise = rng.normal(0.0, sigma, n)
        base = z @ K.T if K is not None else -rho * (obs[slices["u_prev"]] - u0) / env.cfg["du_max"]
        a = np.clip(base + noise, -1.0, 1.0)
        obs_next, r, term, trunc, info = env.step(a.astype(np.float32))
        # Doymada komut etkisizdir: Q'yu gerçekten uygulanan güç değişimiyle eğit.
        a_eff = (obs_next[slices["u_prev"]] - obs[slices["u_prev"]]) / env.cfg["du_max"]
        z_next = state_features(obs_next, slices)[0]
        zs.append(z), acts.append(a_eff), rewards.append(r), z_nexts.append(z_next)
        obs, z = obs_next, z_next
        done = term or trunc
        step += 1
    return {"z": np.array(zs), "a": np.array(acts), "r": np.array(rewards), "z_next": np.array(z_nexts)}


def train(seed: int, cfg: dict, env_cfg: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"seed": seed, "lspi": cfg, "env": env_cfg}, indent=2, ensure_ascii=False), encoding="utf-8")

    rng = np.random.default_rng(seed)
    env = FurnaceEnv(env_cfg)
    slices = env.obs_slices
    val_env = FurnaceEnv(env_cfg)
    val = make_test_set(env_cfg, n=cfg["n_val_scenarios"], base_seed=cfg["val_base_seed"])
    n_z = state_features(env.reset(seed=seed)[0], slices).shape[1]
    q = QuadraticQ(n_z, env.n)

    Z, A, R, Z_next = [], [], [], []
    K = None
    Phi = np.zeros((0, q.n_features))
    best, rows, t0 = -np.inf, [], time.time()
    sigma = cfg["explore_sigma"]

    for it in range(cfg["n_iterations"]):
        n_new = cfg["init_episodes"] if it == 0 else cfg["episodes_per_iter"]
        for _ in range(n_new):
            d = collect(env, K, slices, sigma, cfg["noise_hold"], cfg["rho"], rng)
            Z.append(d["z"]), A.append(d["a"]), R.append(d["r"]), Z_next.append(d["z_next"])
            Phi = np.vstack([Phi, q.features(d["z"], d["a"])])
        r_all, zn_all = np.concatenate(R), np.vstack(Z_next)

        K_iter = K if K is not None else np.zeros((env.n, n_z))
        for _ in range(cfg["inner_iterations"]):
            theta = lstdq_solve(q, Phi, r_all, zn_all, K_iter, cfg["gamma"], cfg["lam"])
            K_new = q.greedy_gain(theta, cfg["min_curv"])
            K_iter = (1.0 - cfg["step_size"]) * K_iter + cfg["step_size"] * K_new
        K = K_iter

        policy = LinearPolicy(K, slices)
        df, _ = evaluate(policy, val_env, val)
        steps = Phi.shape[0]
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
        print(f"[{it:>3}] {steps:>7} adım {row['wall_min']:5.1f} dk  ödül {row['return']:9.1f}  "
              f"RMSE {row['rmse']:6.1f}  düşüş {row['max_drop']:6.1f}  kWh {row['energy_kwh']:6.1f}{star}", flush=True)
        sigma = max(cfg["sigma_min"], sigma * cfg["sigma_decay"])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/lspi.yaml")
    ap.add_argument("--env-config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--name", help="çıktı klasörü adı (varsayılan lspi_seed{seed})")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        key, val = kv.split("=", 1)
        overrides[key] = json.loads(val)
    cfg = merge_config(load_yaml(args.config), overrides)
    env_cfg = load_env_config(args.env_config)
    train(args.seed, cfg, env_cfg, Path("results/models") / (args.name or f"lspi_seed{args.seed}"))


if __name__ == "__main__":
    main()
