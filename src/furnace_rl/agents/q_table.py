"""Tablolu Q-öğrenme: derste işlenen algoritma (sinir ağı yok, fonksiyon yaklaştırma yok).

Her bölgenin kendi Q-tablosu vardır, çünkü üç bölgenin ortak eylem uzayı tablo için çok büyüktür:
    1. Q-tablosunu başlat.
    2-3. ε-greedy ile eylem seç ve uygula.
    4-5. Ödülü ölç, tabloyu güncelle:
         Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

- Eylem: bölgenin hedef güç seviyesi. Ajan 60 s'de bir karar verir. Ortamın ±%10/adım sınırı aynen geçerlidir.
- Durum (bölge başına 10×10 ızgara): bir eksen sıcaklık hatası, diğer eksen fırının yük durumu (`load_class`).
  Yaklaşan yük B1 için önizlemeden, diğer bölgeler için bir önceki bölgenin yükünden gelir.
  `state_mode: error_rate` ile ikinci eksen değişim hızı olur (yük ve önizleme bilgisi yok, karşılaştırma için).
- Ödül: ortamın ödülü bölgelere ayrılır. Eğitimde izleme hatası `err_clip` ile sınırlanabilir: aynı kutudaki çok farklı
  sıcaklıkların kare cezası eylemin etkisini bastırır. Değerlendirme her zaman ortamın asıl ödülüyle yapılır.
- Öğrenme oranı: α(s, a) = max(α_min, α / (1 + ziyaret(s, a) / α_n0)); α_n0 = 0 ise bölüm boyunca doğrusal azalır.

Örnek:
    python -m furnace_rl.agents.q_table --seed 0
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


def load_class(rate: np.ndarray, load: np.ndarray, coming: np.ndarray, b: dict) -> np.ndarray:
    """Yük durumunu 10 sınıfa indirir.
    0-2: yük yok, parti yaklaşmıyor (sıcaklık düşüyor / sabit / yükseliyor)
    3-4: yük yok, küçük / büyük parti yaklaşıyor
    5-6: hafif yük (sıcaklık yükselmiyor / yükseliyor)
    7-8: ağır yük (sıcaklık yükselmiyor / yükseliyor)
    9:   yük var ve büyük parti yaklaşıyor
    """
    R = np.digitize(rate, b["rate"])                # 0: düşüyor, 1: sabit, 2: yükseliyor
    L = np.digitize(load, b["load"])                # 0: yok, 1: hafif, 2: ağır
    C = np.digitize(coming, b["coming"])            # 0: yok, 1: küçük, 2: büyük
    rising = (R == 2).astype(int)
    cls = np.where(C == 0, R, 2 + C)                # yük yokken
    cls = np.where(L == 1, 5 + rising, cls)
    cls = np.where(L == 2, 7 + rising, cls)
    return np.where((L > 0) & (C == 2), 9, cls)


class StateCoder:
    """Gözlemi bölge başına tek bir Q-tablosu satır numarasına çevirir."""

    def __init__(self, cfg: dict, env: FurnaceEnv):
        b = cfg["bins"]
        self.mode = cfg.get("state_mode", "grid")
        self.bins = b
        second = 10 if self.mode == "grid" else len(b["rate"]) + 1
        self.shape = (len(b["error"]) + 1, second)
        self.n_states = int(np.prod(self.shape))
        self.slices = env.obs_slices
        self.err_scale = env.cfg["obs"]["T_err_scale"]
        self.minutes = cfg["decision_steps"] * env.dt / 60.0
        self.reset()

    def reset(self):
        self.prev_e = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, float)
        e = obs[self.slices["T_err"]] * self.err_scale            # °C
        rate = np.zeros_like(e) if self.prev_e is None else (e - self.prev_e) / self.minutes    # °C/dk
        self.prev_e = e.copy()
        load = obs[self.slices["load"]]
        p = obs[self.slices["preview"]]
        coming = np.concatenate([[p[0] * (1.0 - p[2])], load[:-1]])
        second = load_class(rate, load, coming, self.bins) if self.mode == "grid" else np.digitize(rate, self.bins["rate"])
        return np.ravel_multi_index((np.digitize(e, self.bins["error"]), second), self.shape)


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


def learning_rewards(info: dict, env: FurnaceEnv, err_clip: float) -> np.ndarray:
    """Bölge ödülü; izleme cezası |e| ≤ err_clip ile sınırlanır (0: sınır yok). Yalnızca eğitimde kullanılır."""
    r = zone_rewards(info, env)
    if not err_clip:
        return r
    rc = env.cfg["reward"]
    e = (info["T_true"] - info["T_ref"]) / rc["dT_scale"]
    return r + rc["w_track"] * (e ** 2 - np.clip(e, -err_clip, err_clip) ** 2) / env.n


def to_env_action(levels: np.ndarray, obs: np.ndarray, slices: dict, du_max: float) -> np.ndarray:
    """Hedef güç seviyesini ortamın artımlı eylemine çevirir; ±%10/adım sınırı burada korunur."""
    u_prev = np.asarray(obs, float)[slices["u_prev"]]
    return np.clip((levels - u_prev) / du_max, -1.0, 1.0).astype(np.float32)


class QTablePolicy:
    """Açgözlü politika: `k` adımda bir tablodan seviye seçer, arada aynı seviyeyi korur."""

    def __init__(self, q: np.ndarray, coder: StateCoder, levels: np.ndarray, k: int, du_max: float):
        self.q, self.coder, self.levels, self.k, self.du_max = q, coder, levels, k, du_max
        self.reset()

    def reset(self):
        self.t = 0
        self.coder.reset()
        self.target = None

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if self.t % self.k == 0:
            s = self.coder(obs)
            self.target = self.levels[self.q[np.arange(len(s)), s].argmax(axis=1)]
        self.t += 1
        return to_env_action(self.target, obs, self.coder.slices, self.du_max), None


def train(seed: int, cfg: dict, env_cfg: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"seed": seed, "q_table": cfg, "env": env_cfg}, indent=2, ensure_ascii=False), encoding="utf-8")

    rng = np.random.default_rng(seed)
    env = FurnaceEnv(env_cfg)
    val = make_test_set(env_cfg, n=cfg["n_val_scenarios"], base_seed=cfg["val_base_seed"])
    coder = StateCoder(cfg, env)
    eval_coder = StateCoder(cfg, env)
    levels = np.asarray(cfg["levels"], float)
    n_a, k, n = len(levels), cfg["decision_steps"], env.n
    du_max = env.cfg["du_max"]
    q = np.full((n, coder.n_states, n_a), float(cfg["q_init"]))      # 1. adım: Q-tablosunu başlat
    gammas = cfg["gamma"] ** np.arange(k)
    gamma_k = cfg["gamma"] ** k
    zones = np.arange(n)
    visits = np.zeros_like(q)
    n0, a_min, err_clip = cfg.get("alpha_n0", 0), cfg.get("alpha_min", 0.0), cfg.get("err_clip", 0)
    best, rows, t0, steps = -np.inf, [], time.time(), 0

    for ep in range(cfg["n_episodes"]):
        frac = min(1.0, ep / max(1, cfg["eps_decay_episodes"]))
        eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])
        alpha = cfg["alpha"] * (1.0 - cfg.get("alpha_decay", 0.0) * ep / cfg["n_episodes"])
        obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
        coder.reset()
        s = coder(obs)
        done = False
        while not done:
            # 2-3. adım: ε-greedy eylem seçimi, bölge başına bağımsız
            a = q[zones, s].argmax(axis=1)
            explore = rng.random(n) < eps
            a[explore] = rng.integers(n_a, size=int(explore.sum()))
            ret = np.zeros(n)
            for j in range(k):
                env._u_before = env.u_prev.copy()
                obs, _, term, trunc, info = env.step(to_env_action(levels[a], obs, env.obs_slices, du_max))
                ret += gammas[j] * learning_rewards(info, env, err_clip)
                steps += 1
                done = term or trunc
                if done:
                    break
            s2 = coder(obs)
            # 4-5. adım: Q-öğrenme güncellemesi. Bölüm zaman sınırıyla bittiği için son durumdan da ileriye bakılır.
            td = ret + gamma_k * q[zones, s2].max(axis=1) - q[zones, s, a]
            if n0:
                visits[zones, s, a] += 1
                q[zones, s, a] += np.maximum(a_min, cfg["alpha"] / (1.0 + visits[zones, s, a] / n0)) * td
            else:
                q[zones, s, a] += alpha * td
            s = s2

        if (ep + 1) % cfg["eval_every"] == 0 or ep == 0:
            policy = QTablePolicy(q, eval_coder, levels, k, du_max)
            df, _ = evaluate(policy, FurnaceEnv(env_cfg), val)
            row = {"episode": ep, "step": steps, "wall_min": (time.time() - t0) / 60.0, "eps": eps,
                   **{m: float(df[m].mean()) for m in EVAL_KEYS}}
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "evaluations.csv", index=False)
            star = ""
            if row["return"] > best:
                best = row["return"]
                np.savez_compressed(out_dir / "best_model.npz", q=q, levels=levels, k=k)
                star = "  *"
            print(f"[{ep:>5}] {steps:>8} adım {row['wall_min']:5.1f} dk ε={eps:.2f} ödül {row['return']:9.1f}  RMSE "
                  f"{row['rmse']:6.1f}  düşüş {row['max_drop']:6.1f}  aşma {row['overshoot']:5.1f}  "
                  f"kWh {row['energy_kwh']:6.1f}{star}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/q_table.yaml")
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
    train(args.seed, cfg, env_cfg, Path("results/models") / (args.name or f"q_table_seed{args.seed}"))


if __name__ == "__main__":
    main()
