"""Öğrenilen politikayı ısı haritası olarak çizer: her hücrede en büyük Q değerli güç seviyesi.

Çerçeveli hücreler, 20 test senaryosunda zamanın en az %2'sinin geçtiği hücrelerdir.

Örnek:
    python scripts/plot_policy.py --run results/models/q_son_seed0 --zone 0 --out results/figures/politika_b1.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from furnace_rl.envs.furnace_env import FurnaceEnv
from furnace_rl.envs.scenarios import load_scenarios
from furnace_rl.evaluate import load_run_policy

ERR = ["< −100", "−100…−60", "−60…−30", "−30…−15", "−15…−8", "−8…−3", "−3…3", "3…8", "8…20", "> 20"]
LOAD = ["yüksüz\ndüşüyor", "yüksüz\nsabit", "yüksüz\nyükseliyor", "küçük parti\ngeliyor", "büyük parti\ngeliyor",
        "hafif\nyükselmiyor", "hafif\nyükseliyor", "ağır\nyükselmiyor", "ağır\nyükseliyor", "yük +\nbüyük parti"]
ZONES = ["B1 (ön ısıtma)", "B2 (ısıtma)", "B3 (tutma)"]


def occupancy(run: str, zone: int, scenarios: str) -> np.ndarray:
    """Açgözlü politikayla test senaryolarında her hücrede geçen zamanın yüzdesi."""
    env = FurnaceEnv()
    policy = load_run_policy(run, env)
    coder = policy.coder
    counts = np.zeros(coder.n_states)

    class Counting:
        def __call__(self, obs):
            s = coder(obs)
            counts[s[zone]] += 1
            return s

        def __getattr__(self, name):
            return getattr(coder, name)

    policy.coder = Counting()
    for sc in load_scenarios(scenarios):
        obs, _ = env.reset(seed=sc["seed"], options={"scenario": sc["schedule"]})
        policy.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    return (100 * counts / counts.sum()).reshape(10, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/models/q_son_seed0")
    ap.add_argument("--zone", type=int, default=0)
    ap.add_argument("--scenarios", default="configs/test_scenarios.json")
    ap.add_argument("--min-share", type=float, default=2.0, help="çerçeve için en az zaman yüzdesi")
    ap.add_argument("--out", default="results/figures/politika_b1.png")
    args = ap.parse_args()

    z = np.load(f"{args.run}/best_model.npz")
    levels = z["levels"]
    greedy = levels[z["q"][args.zone].argmax(axis=1)].reshape(10, 10)
    share = occupancy(args.run, args.zone, args.scenarios)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    im = ax.imshow(greedy, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(10):
        for j in range(10):
            v = greedy[i, j]
            ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=10,
                    color="white" if v >= 0.6 else "#1B1F27", fontweight="bold" if share[i, j] >= args.min_share else None)
            if share[i, j] >= args.min_share:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec="#1B1F27", lw=2))
    ax.set_xticks(range(10), LOAD, fontsize=8.5)
    ax.set_yticks(range(10), ERR, fontsize=9)
    ax.set_xlabel("yük durumu (sütun)")
    ax.set_ylabel("sıcaklık hatası T − T_hedef [°C] (satır)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    covered = share[share >= args.min_share].sum()
    n_cells = int((share >= args.min_share).sum())
    ax.set_title(f"{ZONES[args.zone]} için öğrenilen politika: her hücrede seçilen güç seviyesi\n"
                 f"çerçeveli {n_cells} hücre testte zamanın %{covered:.0f}'ini kapsar", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("güç seviyesi")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"{args.out} yazıldı; çerçeveli hücre {n_cells}, zamanın %{covered:.1f}'i")


if __name__ == "__main__":
    main()
