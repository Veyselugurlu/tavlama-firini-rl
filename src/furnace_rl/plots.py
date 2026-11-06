"""Rapor grafikleri."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ZONE_NAMES = ["Bölge 1 (ön ısıtma)", "Bölge 2 (ısıtma)", "Bölge 3 (tutma)"]
COLORS = ["#2a6fdb", "#e0622b", "#2a9d5c", "#8a4fd0"]


def plot_episode(trajs: dict[str, dict], path: str | Path, title: str = "") -> None:
    """Her bölgenin sıcaklığı (politikalar üst üste) + güç komutları. `trajs`: etiket → yörünge."""
    first = next(iter(trajs.values()))
    n = first["T_true"].shape[1]
    fig, axes = plt.subplots(n + 1, 1, figsize=(10, 2.4 * (n + 1)), sharex=True)
    t_min = first["t"] / 60.0

    for i in range(n):
        ax = axes[i]
        ax.plot(t_min, first["T_ref"][:, i], "k--", lw=1, label="hedef")
        for j, (label, tr) in enumerate(trajs.items()):
            ax.plot(tr["t"] / 60.0, tr["T_true"][:, i], color=COLORS[j % len(COLORS)], lw=1.4, label=label)
        ax.set_ylabel("°C")
        ax.set_title(ZONE_NAMES[i] if i < len(ZONE_NAMES) else f"Bölge {i + 1}", fontsize=10, loc="left")
        ax.grid(alpha=0.3)

    ax = axes[-1]
    styles = ["-", "--", ":", "-."]
    for j, (label, tr) in enumerate(trajs.items()):
        for i in range(n):
            ax.plot(tr["t"] / 60.0, tr["u_eff"][:, i], color=COLORS[i % len(COLORS)],
                    ls=styles[j % len(styles)], lw=1.2,
                    label=f"{label} – B{i + 1}" if len(trajs) > 1 else f"B{i + 1}")
    ax.set_ylabel("güç oranı")
    ax.set_xlabel("zaman [dk]")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=n, loc="upper right")

    for ax in axes:
        for ta in first["arrivals"]:
            ax.axvline(ta / 60.0, color="gray", alpha=0.5, lw=0.8)
    axes[0].legend(fontsize=8, loc="lower right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_learning_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path,
                         ylabel: str = "bölüm ödülü", reference: tuple[str, float] | None = None) -> None:
    """`curves`: etiket → (adımlar, [seed × nokta] değerler). Ortalama ± std bandı.
    `reference`: (etiket, değer) yatay çizgi, ör. PID."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, (label, (x, Y)) in enumerate(curves.items()):
        Y = np.atleast_2d(Y)
        mu, sd = Y.mean(0), Y.std(0)
        c = COLORS[j % len(COLORS)]
        ax.plot(x, mu, color=c, label=f"{label} (n={Y.shape[0]} seed)")
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.2)
    if reference is not None:
        ax.axhline(reference[1], color="k", ls="--", lw=1, label=reference[0])
    ax.set_xlabel("ortam adımı")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bars(groups: dict, metrics: list[str], titles: list[str], path: str | Path) -> None:
    """`groups`: etiket → senaryo bazında metrik DataFrame'i. Her metrik için ortalama ± std çubuğu."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.6))
    for ax, m, title in zip(np.atleast_1d(axes), metrics, titles):
        labels = list(groups)
        mu = [groups[g][m].mean() for g in labels]
        sd = [groups[g][m].std() for g in labels]
        ax.bar(labels, mu, yerr=sd, color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
               capsize=4, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def to_markdown(df) -> str:
    """tabulate bağımlılığı olmadan basit Markdown tablo."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines) + "\n"
