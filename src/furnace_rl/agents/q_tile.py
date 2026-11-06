"""Tile coding'li Q-öğrenme (Sutton & Barto, Bölüm 10; sinir ağı yok).

Her bölgede bağımsız bir ajan vardır:
    Q_i(s_i, a) = Σ_{t=1..n_tilings} w_i[t, kare_t(s_i), a]      (doğrusal, seyrek ikili özellikler)
    w ← w + (α / n) [ r_i + γ max_a' Q_i(s_i', a') − Q_i(s_i, a) ]

- Eylem: ayrık güç değişimi, `decision_steps` kontrol adımı boyunca tutulur (varsayılan 60 s).
- Durum (5 boyut): hata, türev, u_prev − u_eff, bölgedeki yük, yaklaşan yük.
  Yaklaşan yük B1 için önizleme m·(1 − τ), diğer bölgeler için bir önceki bölgenin yüküdür.
- Ödül: ortamın ödülü bölgelere ayrılır (Σ r_i = r, güvenlik cezası hariç bölge bazlıdır).

Örnek:
    python -m furnace_rl.agents.q_tile --seed 0
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


class TileCoder:
    """Her boyutu [lo, hi] aralığında `bins` kareye böler; n tiling asimetrik kaydırılmıştır."""

    def __init__(self, lows, highs, bins, n_tilings: int):
        self.lows = np.asarray(lows, float)
        self.widths = (np.asarray(highs, float) - self.lows) / (np.asarray(bins) - 1)
        self.bins = np.asarray(bins, int)
        self.n_tilings = n_tilings
        # Sutton & Barto: tek sayılı yer değiştirme vektörü (1, 3, 5, ...) ile asimetrik kaydırma.
        disp = 2 * np.arange(len(self.bins)) + 1
        self.offsets = (np.arange(n_tilings)[:, None] * disp / n_tilings) % 1.0
        self.strides = np.concatenate([[1], np.cumprod(self.bins[:-1])])
        self.tiles_per_tiling = int(np.prod(self.bins))
        self.base = np.arange(n_tilings) * self.tiles_per_tiling
        self.size = self.tiles_per_tiling * n_tilings

    def tiles(self, s: np.ndarray) -> np.ndarray:
        scaled = (np.clip(s, self.lows, self.lows + self.widths * (self.bins - 1)) - self.lows) / self.widths
        idx = np.minimum((scaled + self.offsets).astype(int), self.bins - 1)
        return self.base + idx @ self.strides


def zone_states(obs: np.ndarray, slices: dict) -> np.ndarray:
    """Bölge başına durum: satır = bölge, sütun = (hata, türev, gecikme, yük, yaklaşan yük)."""
    # Karekök ölçek: hedefe yakın bölgede ince, uzakta kaba kareler.
    e = np.sign(obs[slices["T_err"]]) * np.sqrt(np.abs(obs[slices["T_err"]]))
    d = np.sign(obs[slices["dT"]]) * np.sqrt(np.abs(obs[slices["dT"]]))
    lag = obs[slices["u_prev"]] - obs[slices["u_eff"]]
    load = obs[slices["load"]]
    p = obs[slices["preview"]]
    coming = np.concatenate([[p[0] * (1.0 - p[2])], load[:-1]])
    return np.column_stack([e, d, lag, load, coming])


def zone_rewards(info: dict, env: FurnaceEnv) -> np.ndarray:
    """Ortam ödülünün bölge payları: toplamları ortamın ödülüyle aynı terimlerden oluşur."""
    rc = env.cfg["reward"]
    n = env.n
    t = info["reward_terms"]
    e = (info["T_true"] - info["T_ref"]) / rc["dT_scale"]
    energy_share = env.P_max * info["u_eff"]
    energy_share = energy_share / max(energy_share.sum(), 1e-9) * t["energy"]
    smooth = np.abs(info["u_cmd"] - env._u_before) if hasattr(env, "_u_before") else np.zeros(n)
    return -(rc["w_track"] * e ** 2 / n + rc["w_energy"] * energy_share + rc["w_smooth"] * smooth / n) - t["safety"] / n


class TileQPolicy:
    """Açgözlü politika: `decision_steps` adımda bir karar verir, arada eylemi tutar."""

    def __init__(self, w: np.ndarray, coder: TileCoder, actions: np.ndarray, slices: dict, decision_steps: int):
        self.w, self.coder, self.actions, self.slices = w, coder, actions, slices
        self.k = decision_steps
        self.reset()

    def reset(self):
        self.t = 0
        self.a = np.zeros(self.w.shape[0], dtype=np.float32)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if self.t % self.k == 0:
            S = zone_states(np.asarray(obs, float), self.slices)
            for i in range(S.shape[0]):
                q = self.w[i, self.coder.tiles(S[i])].sum(axis=0)
                self.a[i] = self.actions[int(np.argmax(q))]
        self.t += 1
        return self.a.copy(), None


def make_coder(cfg: dict) -> TileCoder:
    c = cfg["tiles"]
    return TileCoder(c["low"], c["high"], c["bins"], cfg["n_tilings"])


def train(seed: int, cfg: dict, env_cfg: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"seed": seed, "q_tile": cfg, "env": env_cfg}, indent=2, ensure_ascii=False), encoding="utf-8")

    rng = np.random.default_rng(seed)
    env = FurnaceEnv(env_cfg)
    slices = env.obs_slices
    val = make_test_set(env_cfg, n=cfg["n_val_scenarios"], base_seed=cfg["val_base_seed"])
    coder = make_coder(cfg)
    actions = np.asarray(cfg["actions"], dtype=np.float32)
    n_a, k = len(actions), cfg["decision_steps"]
    # Tüm gerçek Q değerleri negatif; 0'dan başlamak denenmemiş eylemleri her zaman en iyi gösterir.
    # Başlangıç, tipik bir karar değeri civarında seçilir (hiperparametre).
    w = np.full((env.n, coder.size, n_a), cfg["q_init"] / cfg["n_tilings"], dtype=np.float32)
    alpha = cfg["alpha"] / cfg["n_tilings"]
    gamma_k = cfg["gamma"] ** k
    gammas = cfg["gamma"] ** np.arange(k)
    best, rows, t0, steps = -np.inf, [], time.time(), 0

    for ep in range(cfg["n_episodes"]):
        frac = min(1.0, ep / max(1, cfg["eps_decay_episodes"]))
        eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])
        a_lr = alpha * (1.0 - cfg["alpha_decay"] * ep / cfg["n_episodes"])
        obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
        done = False
        while not done:
            S = zone_states(obs, slices)
            tiles = [coder.tiles(S[i]) for i in range(env.n)]
            a_idx = np.array([
                rng.integers(n_a) if rng.random() < eps else int(np.argmax(w[i, tiles[i]].sum(axis=0)))
                for i in range(env.n)])
            a = actions[a_idx]
            ret = np.zeros(env.n)
            for j in range(k):
                env._u_before = env.u_prev.copy()
                obs, _, term, trunc, info = env.step(a)
                ret += gammas[j] * zone_rewards(info, env)
                steps += 1
                done = term or trunc
                if done:
                    break
            S2 = zone_states(obs, slices)
            for i in range(env.n):
                q_next = w[i, coder.tiles(S2[i])].sum(axis=0).max()
                td = ret[i] + gamma_k * q_next - w[i, tiles[i], a_idx[i]].sum()
                w[i, tiles[i], a_idx[i]] += a_lr * td

        if (ep + 1) % cfg["eval_every"] == 0 or ep == 0:
            policy = TileQPolicy(w, coder, actions, slices, k)
            df, _ = evaluate(policy, FurnaceEnv(env_cfg), val)
            row = {"episode": ep, "step": steps, "wall_min": (time.time() - t0) / 60.0, "eps": eps,
                   **{m: float(df[m].mean()) for m in EVAL_KEYS}}
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "evaluations.csv", index=False)
            star = ""
            if row["return"] > best:
                best = row["return"]
                np.savez_compressed(out_dir / "best_model.npz", w=w, actions=actions, k=k)
                star = "  *"
            print(f"[{ep:>5}] {steps:>8} adım {row['wall_min']:5.1f} dk ε={eps:.2f} ödül {row['return']:9.1f}  RMSE "
                  f"{row['rmse']:6.1f}  düşüş {row['max_drop']:6.1f}  aşma {row['overshoot']:5.1f}  "
                  f"kWh {row['energy_kwh']:6.1f}{star}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/q_tile.yaml")
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
    train(args.seed, cfg, env_cfg, Path("results/models") / (args.name or f"q_tile_seed{args.seed}"))


if __name__ == "__main__":
    main()
