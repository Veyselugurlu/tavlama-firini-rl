"""README için animasyon: aynı test senaryosunda PID ve Q-tablosu fırını yan yana yönetir.

Üstte iki fırın şeması (bölge rengi hedeften sapma, koyu bloklar partiler, turuncu çubuk ısıtıcı gücü),
altta B1 sıcaklığı. Çıktı: results/figures/firin_animasyon.gif

Örnek:
    python scripts/make_gif.py --scenario test_12 --run results/models/q_son_seed0
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from furnace_rl.baselines.pid import PIDPolicy
from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import load_scenarios
from furnace_rl.evaluate import load_run_policy

INK, MUTED = "#1B1F27", "#5F6570"
COLORS = {"PID": "#2a6fdb", "Q-tablosu": "#e0622b"}
ZONES = ["B1 ön ısıtma", "B2 ısıtma", "B3 tutma"]


def record(policy, scenario: dict) -> dict:
    """Senaryoyu koşturur, her adımda sıcaklık, ısıtıcı gücü ve parti konumlarını saklar."""
    env = FurnaceEnv()
    obs, _ = env.reset(seed=scenario["seed"], options={"scenario": scenario["schedule"]})
    policy.reset()
    rec = {"t": [], "T": [], "u": [], "bx": [], "bm": [], "next": []}
    done = False
    while not done:
        action, _ = policy.predict(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        rec["t"].append(env.t / 60.0)
        rec["T"].append(env.T.copy())
        rec["u"].append(env.u_eff.copy())
        rec["bx"].append(env.b_x.copy())
        rec["bm"].append(env.b_m.copy())
        nxt = env.pending[0] if env.pending else None
        rec["next"].append(None if nxt is None else ((nxt["t_arrival"] - env.t) / 60.0, nxt["mass"]))
    rec["T"], rec["u"], rec["t"] = np.array(rec["T"]), np.array(rec["u"]), np.array(rec["t"])
    rec["T_ref"], rec["L"], rec["arrivals"] = env.T_ref.copy(), env.zone_length, list(env.arrivals)
    return rec


def draw_furnace(ax, rec, k, name, norm, cmap):
    ax.cla()
    L, n = rec["L"], len(rec["T_ref"])
    for i in range(n):
        err = rec["T"][k, i] - rec["T_ref"][i]
        ax.add_patch(Rectangle((i * L, 0.25), L, 1.0, fc=cmap(norm(err)), ec="white", lw=3))
        ink = "white" if abs(err) > 30 else INK          # koyu zeminde okunur yazı
        ax.text(i * L + 0.25, 1.08, f"{ZONES[i]}", fontsize=9, color=ink, va="top")
        ax.text(i * L + 0.25, 0.80, f"{rec['T'][k, i]:.0f} °C  ({err:+.0f})", fontsize=11, color=ink,
                va="center", fontweight="bold")
        ax.add_patch(Rectangle((i * L + 0.25, 0.32), (L - 0.5) * rec["u"][k, i], 0.13, fc="#C2410C", ec="none"))
        ax.add_patch(Rectangle((i * L + 0.25, 0.32), L - 0.5, 0.13, fc="none", ec="white" if abs(err) > 30 else MUTED, lw=0.8))
    for x, m in zip(rec["bx"][k], rec["bm"][k]):
        h = 0.12 + 0.28 * m / 1500.0
        ax.add_patch(Rectangle((x - 0.35, 0.47), 0.7, h, fc="#3D434E", ec="white", lw=1))
    ax.text(-0.3, 0.75, name, fontsize=13, fontweight="bold", color=COLORS[name], ha="right", va="center")
    ax.set_xlim(-2.6, n * L + 0.2)
    ax.set_ylim(0.2, 1.3)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="test_12")
    ap.add_argument("--run", default="results/models/q_son_seed0")
    ap.add_argument("--every", type=int, default=5, help="kaç kontrol adımında bir kare")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="results/figures/firin_animasyon.gif")
    args = ap.parse_args()

    sc = next(s for s in load_scenarios("configs/test_scenarios.json") if s["name"] == args.scenario)
    env = FurnaceEnv()
    recs = {"PID": record(PIDPolicy.from_yaml(env, "configs/pid.yaml"), sc),
            "Q-tablosu": record(load_run_policy(args.run, env), sc)}
    frames = list(range(0, len(recs["PID"]["t"]), args.every)) + [len(recs["PID"]["t"]) - 1]

    cmap, norm = plt.get_cmap("RdBu_r"), TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    fig = plt.figure(figsize=(10.5, 6.0), dpi=80)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.6], hspace=0.25, left=0.04, right=0.97, top=0.88, bottom=0.15)
    ax_p, ax_q, ax_t = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    title = fig.text(0.04, 0.95, "", fontsize=13, fontweight="bold", color=INK)
    sub = fig.text(0.04, 0.905, "", fontsize=10, color=MUTED)
    fig.text(0.04, 0.015, "bölge rengi: mavi = hedefin altında, kırmızı = üstünde · koyu blok = parti · turuncu çubuk = ısıtıcı gücü",
             fontsize=9, color=MUTED, ha="left")

    t_all = recs["PID"]["t"]
    ref = recs["PID"]["T_ref"][0]
    ax_t.axhline(ref, color=INK, ls="--", lw=1)
    ax_t.text(t_all[-1], ref + 3, "hedef 750 °C", ha="right", va="bottom", fontsize=9, color=INK)
    lines = {name: ax_t.plot([], [], color=COLORS[name], lw=2, label=name)[0] for name in recs}
    lo = min(r["T"][:, 0].min() for r in recs.values()) - 10
    hi = max(r["T"][:, 0].max() for r in recs.values()) + 15
    ax_t.set_xlim(0, t_all[-1])
    ax_t.set_ylim(lo, hi)
    ax_t.set_xlabel("zaman [dk]")
    ax_t.set_ylabel("B1 sıcaklığı [°C]")
    ax_t.legend(loc="lower left", frameon=False, ncol=2)
    for s in ("top", "right"):
        ax_t.spines[s].set_visible(False)
    ax_t.grid(alpha=0.25)
    arrival_lines = []

    def update(k):
        draw_furnace(ax_p, recs["PID"], k, "PID", norm, cmap)
        draw_furnace(ax_q, recs["Q-tablosu"], k, "Q-tablosu", norm, cmap)
        for name, line in lines.items():
            line.set_data(t_all[: k + 1], recs[name]["T"][: k + 1, 0])
        for ta in recs["PID"]["arrivals"]:
            if ta / 60.0 <= t_all[k] and ta not in arrival_lines:
                ax_t.axvline(ta / 60.0, color=MUTED, alpha=0.5, lw=0.8)
                arrival_lines.append(ta)
        title.set_text(f"Aynı senaryoda PID ve Q-tablosu  ·  t = {t_all[k]:.0f} dk")
        nxt = recs["PID"]["next"][k]
        sub.set_text("Önizleme: sırada parti yok" if nxt is None
                     else f"Önizleme: {nxt[1]:.0f} kg'lık parti {max(nxt[0], 0):.0f} dk sonra fırına giriyor")
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"{args.out} yazıldı: {len(frames)} kare")


if __name__ == "__main__":
    main()
