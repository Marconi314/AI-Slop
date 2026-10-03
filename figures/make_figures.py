#!/usr/bin/env python3
"""
Generates the static figures used by euler-identity.html.

Run from the repo root (the folder that contains euler-identity.html):

    python3 figures/make_figures.py

Output goes to the images/ folder. Requires: numpy, matplotlib.

The figures are drawn on the same dark "card" color the page uses, so they
blend into the page. Colors follow the page's convention:
    blue   = real part / cosine / in-phase
    orange = imaginary part / sine / quadrature
    green  = the complex number itself / phasor / sum
    yellow = highlight / harmonics
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

# ---- palette (matches the CSS variables in the page) -----------------------
CARD   = "#121a2e"   # figure background = page card color
INK    = "#e8ecf5"   # primary text
MUTED  = "#9aa6c0"   # secondary text / axes
GRID   = "#26314d"   # faint grid
BLUE   = "#3987e5"   # real / cos
ORANGE = "#d95926"   # imaginary / sin
GREEN  = "#199e70"   # phasor / sum
YELLOW = "#c98500"   # harmonics / highlight
VIOLET = "#9085e9"
MAGENTA = "#d55181"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": CARD,
    "savefig.facecolor": CARD,
    "axes.facecolor": CARD,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "mathtext.fontset": "dejavusans",
    "lines.linewidth": 2.2,
    "legend.frameon": False,
    "legend.labelcolor": INK,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def clean(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. The complex plane: e^{j theta} = cos(theta) + j sin(theta)
# ---------------------------------------------------------------------------
def fig_euler_plane():
    th = np.deg2rad(50)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.4, ls=(0, (4, 4)))
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)

    x, y = np.cos(th), np.sin(th)
    # projections
    ax.plot([x, x], [0, y], color=ORANGE, lw=2.6)
    ax.plot([0, x], [0, 0], color=BLUE, lw=2.6)
    ax.plot([0, x], [y, y], color=ORANGE, lw=1, ls=":")
    ax.plot([x, x], [0, 0], color=BLUE)
    # the phasor
    ax.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle="-|>", mutation_scale=18,
                                 color=GREEN, lw=3))
    ax.plot([x], [y], "o", color=GREEN, ms=9, mec=CARD, mew=2)
    # angle arc
    ax.add_patch(Arc((0, 0), 0.5, 0.5, theta1=0, theta2=50, color=YELLOW, lw=2.2))
    ax.text(0.33, 0.1, r"$\theta$", color=YELLOW, fontsize=15, ha="center", va="center")

    ax.text(x / 2, -0.09, r"$\cos\theta$", color=BLUE, ha="center", va="top", fontsize=14)
    ax.text(x + 0.04, y / 2, r"$\sin\theta$", color=ORANGE, ha="left", va="center", fontsize=14)
    ax.text(x / 2 - 0.12, y / 2 + 0.1, r"$e^{j\theta}$", color=GREEN, ha="right", va="center", fontsize=16)
    ax.text(1.08, 0.04, "Re", color=MUTED, fontsize=11)
    ax.text(0.03, 1.1, "Im", color=MUTED, fontsize=11)
    ax.text(1.07, -0.07, "1", color=MUTED, ha="center", va="top", fontsize=10)
    ax.text(-0.06, 1.07, "$j$", color=MUTED, ha="right", va="center", fontsize=11)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig_euler_plane.png")


# ---------------------------------------------------------------------------
# 2. Taylor polynomials converging to cos(theta)
# ---------------------------------------------------------------------------
def fig_taylor_cos():
    th = np.linspace(-2 * np.pi - 0.6, 2 * np.pi + 0.6, 1200)
    from math import factorial

    def partial(order):
        s = np.zeros_like(th)
        for k in range(0, order + 1, 2):
            s += (-1) ** (k // 2) * th ** k / factorial(k)
        return s

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    clean(ax)
    ax.plot(th, np.cos(th), color=INK, lw=3, label=r"$\cos\theta$", zorder=5)
    for order, col in [(2, MAGENTA), (6, VIOLET), (10, YELLOW), (16, GREEN)]:
        ax.plot(th, partial(order), color=col, lw=2, label=f"degree {order}")
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlim(th[0], th[-1])
    ax.set_xlabel(r"$\theta$  (radians)")
    ax.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_xticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    ax.legend(loc="upper center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 1.14))
    save(fig, "fig_taylor_cos.png")


# ---------------------------------------------------------------------------
# 3. The special points on the unit circle -> Euler's identity
# ---------------------------------------------------------------------------
def fig_identity_points():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.4, ls=(0, (4, 4)))
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)

    pts = [
        (0, r"$e^{j0}=1$", (1, 0), (12, -26), "left"),
        (np.pi / 2, r"$e^{j\pi/2}=j$", (0, 1), (12, 10), "left"),
        (np.pi, r"$e^{j\pi}=-1$", (-1, 0), (-12, -26), "right"),
        (3 * np.pi / 2, r"$e^{j3\pi/2}=-j$", (0, -1), (12, -20), "left"),
    ]
    for ang, label, (px, py), off, ha in pts:
        hi = abs(ang - np.pi) < 1e-9
        col = YELLOW if hi else GREEN
        ax.add_patch(FancyArrowPatch((0, 0), (px, py), arrowstyle="-|>", mutation_scale=14,
                                     color=col, lw=2.4 if hi else 1.8))
        ax.plot([px], [py], "o", color=col, ms=9 if hi else 7, mec=CARD, mew=2)
        ax.annotate(label, (px, py), textcoords="offset points", xytext=off, ha=ha,
                    color=col, fontsize=14 if hi else 12)
    # the half turn
    ax.add_patch(Arc((0, 0), 0.7, 0.7, theta1=0, theta2=180, color=YELLOW, lw=2))
    ax.text(0, 0.46, r"half turn: $\pi$ rad", color=YELLOW, ha="center", fontsize=11)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig_identity_points.png")


# ---------------------------------------------------------------------------
# 4. A real cosine = two counter-rotating phasors -> two-sided spectrum
# ---------------------------------------------------------------------------
def fig_cos_spectrum():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.7),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    f0 = 1.0
    t = np.linspace(-1.5, 1.5, 800)
    clean(a1)
    a1.plot(t, np.cos(2 * np.pi * f0 * t), color=BLUE)
    a1.set_title(r"time domain:  $x(t)=\cos(2\pi f_0 t)$")
    a1.set_xlabel("t")
    a1.set_ylim(-1.4, 1.4)

    clean(a2, grid=False)
    for f in (-f0, f0):
        a2.vlines(f, 0, 0.5, color=GREEN, lw=4)
        a2.plot([f], [0.5], "o", color=GREEN, ms=9, mec=CARD, mew=2)
    a2.text(f0, 0.58, r"$\frac{1}{2}e^{+j2\pi f_0 t}$", color=GREEN, ha="center", fontsize=12)
    a2.text(-f0, 0.58, r"$\frac{1}{2}e^{-j2\pi f_0 t}$", color=GREEN, ha="center", fontsize=12)
    a2.axhline(0, color=MUTED, lw=1)
    a2.set_xlim(-2.6, 2.6)
    a2.set_ylim(0, 0.8)
    a2.set_xticks([-f0, 0, f0])
    a2.set_xticklabels([r"$-f_0$", "0", r"$+f_0$"])
    a2.set_yticks([])
    a2.set_xlabel("frequency")
    a2.set_title("frequency domain: two phasors")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_cos_spectrum.png")


# ---------------------------------------------------------------------------
# 5. Gibbs phenomenon: partial Fourier sums of a square wave
# ---------------------------------------------------------------------------
def fig_gibbs():
    t = np.linspace(-1.0, 1.0, 4000)
    sq = np.sign(np.sin(2 * np.pi * t))
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), sharex=True, sharey=True)
    for ax, n_h in zip(axes.flat, (1, 3, 9, 49)):
        clean(ax)
        s = np.zeros_like(t)
        for k in range(1, n_h + 1, 2):
            s += (4 / (np.pi * k)) * np.sin(2 * np.pi * k * t)
        ax.plot(t, sq, color=MUTED, lw=1.2, ls=(0, (3, 3)))
        ax.plot(t, s, color=GREEN, lw=1.8)
        ax.set_title(f"harmonics up to {n_h}", fontsize=11)
        ax.set_ylim(-1.5, 1.5)
        if n_h == 49:
            ax.annotate("≈ 9% overshoot\n(never goes away)", (0.012, 1.17), (0.27, 0.35),
                        color=YELLOW, fontsize=10, va="center", ha="center",
                        arrowprops=dict(arrowstyle="->", color=YELLOW))
    for ax in axes[1]:
        ax.set_xlabel("time (in periods)")
    fig.tight_layout()
    save(fig, "fig_gibbs.png")


# ---------------------------------------------------------------------------
# 6. Time-frequency duality pairs
# ---------------------------------------------------------------------------
def fig_duality_pairs():
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4))
    t = np.linspace(-3, 3, 1500)
    f = np.linspace(-4, 4, 1500)

    # rect -> sinc
    clean(axes[0, 0])
    axes[0, 0].plot(t, (np.abs(t) <= 0.5).astype(float), color=BLUE)
    axes[0, 0].set_title("rectangular pulse (width 1)")
    clean(axes[0, 1])
    axes[0, 1].plot(f, np.sinc(f), color=GREEN)
    axes[0, 1].set_title(r"its spectrum:  $\mathrm{sinc}(f)$")

    # gaussian -> gaussian
    sig = 0.5
    clean(axes[1, 0])
    axes[1, 0].plot(t, np.exp(-t ** 2 / (2 * sig ** 2)), color=BLUE)
    axes[1, 0].set_title("Gaussian pulse")
    clean(axes[1, 1])
    axes[1, 1].plot(f, np.exp(-2 * np.pi ** 2 * sig ** 2 * f ** 2), color=GREEN)
    axes[1, 1].set_title("its spectrum is also a Gaussian")

    for a in axes[:, 0]:
        a.set_xlabel("t")
    for a in axes[:, 1]:
        a.set_xlabel("f")
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    save(fig, "fig_duality_pairs.png")


# ---------------------------------------------------------------------------
# 7. Sampling: spectrum replicas and aliasing
# ---------------------------------------------------------------------------
def fig_aliasing_fold():
    fs = 10.0
    f0 = 7.0   # above fs/2 = 5 -> aliases to 3
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    clean(ax, grid=False)
    ax.axhline(0, color=MUTED, lw=1)
    # Nyquist band shading
    ax.axvspan(-fs / 2, fs / 2, color=GRID, alpha=0.55, lw=0)
    ax.text(0, 1.62, "baseband: all you can see after sampling", color=MUTED,
            ha="center", fontsize=10)

    for k in range(-2, 3):
        for sgn in (+1, -1):
            f = sgn * f0 + k * fs
            if abs(f) > 16:
                continue
            true = (k == 0)
            col = BLUE if true else MAGENTA
            ax.vlines(f, 0, 1, color=col, lw=3.4 if true else 2.2,
                      alpha=1 if true else 0.9)
            ax.plot([f], [1], "o", color=col, ms=8, mec=CARD, mew=2)
    # the alias that lands in baseband
    ax.annotate("true tone\n+7 Hz", (7, 1), (7.0, 1.2), color=BLUE, ha="center",
                fontsize=10, va="bottom")
    ax.annotate("alias lands\nat 3 Hz", (3, 1), (3.0, 1.2), color=MAGENTA, ha="center",
                fontsize=10, va="bottom")
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 1.78)
    ax.set_yticks([])
    ticks = [-15, -10, -5, 0, 5, 10, 15]
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$-1.5f_s$", r"$-f_s$", r"$-f_s/2$", "0", r"$f_s/2$", r"$f_s$", r"$1.5f_s$"])
    ax.set_xlabel(r"frequency  ($f_s = 10$ Hz;  tone at 7 Hz is above $f_s/2$)")
    save(fig, "fig_aliasing_fold.png")


if __name__ == "__main__":
    fig_euler_plane()
    fig_taylor_cos()
    fig_identity_points()
    fig_cos_spectrum()
    fig_gibbs()
    fig_duality_pairs()
    fig_aliasing_fold()
    print("done")
