"""Schematic animation: carry-last vs. InfiLoop state update over loop steps (illustrative numbers)."""
import os, subprocess, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('OUT', HERE / 'out'))
FR = HERE / 'frames_mech'
shutil.rmtree(FR, ignore_errors=True); FR.mkdir(parents=True); OUT.mkdir(exist_ok=True)

# Illustrative candidate stream: improves, is solved at step 5, then noisy at steps 6 and 8.
rng = np.random.default_rng(3)
order = rng.permutation(16)
Q = [6, 9, 12, 14, 16, 10, 16, 9]
C = []
for t, k in enumerate(Q):
    c = np.zeros(16)
    if t in (5, 7):
        c[:] = 1; c[rng.choice(16, 16 - k, replace=False)] = 0
    else:
        c[order[:k]] = 1
    C.append(c)
T = len(Q)
BETA = .55
E = [np.exp(4 * (k / 16 - .5)) for k in Q]          # content score tracks correctness
H, W, G = [], [], []                                  # InfiLoop state, weights over past, step size
N, D = np.zeros(16), 0.0
for t in range(T):
    N = BETA * N + E[t] * C[t]; D = BETA * D + E[t]
    H.append(N / D); G.append(E[t] / D)
    W.append(np.array([E[i] * BETA ** (t - i) for i in range(t + 1)]) / D)

GOOD, BAD, INK, FAINT, BG = '#3A5BD9', '#F07167', '#1F2328', '#E4E7EC', '#FFFFFF'
MUTED = INK
BLUE, ORANGE = '#3A5BD9', '#F08A4B'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})

def mix(v):
    a, b = np.array(matplotlib.colors.to_rgb(BAD)), np.array(matplotlib.colors.to_rgb(GOOD))
    return tuple(a + (b - a) * float(np.clip(v, 0, 1)))

def grid(ax, x, y, s, vals, alpha=1.0, decode=False, mono=None):
    cs = s / 4
    for j, v in enumerate(vals):
        r, c = divmod(j, 4)
        col = mono if mono else mix(1.0 if v > .5 else 0.0) if decode else mix(v)
        ax.add_patch(Rectangle((x + c * cs, y + (3 - r) * cs), cs * .9, cs * .9, fc=col, ec='none', alpha=alpha))

def frame(idx, t, a_new, a_upd, final=0.0):
    """t: current step (0-based); a_new: fade-in of candidate t; a_upd: progress of state/weight update."""
    fig = plt.figure(figsize=(10, 4.6), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(4, 50); ax.axis('off')
    ax.text(3, 46, 'Carrying state across loops', fontsize=18, weight='bold', va='center')
    ax.text(3, 41.5, f'loop step {t + 1}', fontsize=12, color=MUTED, va='center')
    ax.text(17, 36.5, 'candidates from each loop', fontsize=10.5, color=MUTED, va='center')
    # legend
    for k, (lab, col) in enumerate([('correct cell', GOOD), ('wrong cell', BAD)]):
        ax.add_patch(Rectangle((62 + k * 17, 40.8), 1.4, 1.4, fc=col, ec='none'))
        ax.text(64.2 + k * 17, 41.5, lab, fontsize=10.5, color=MUTED, va='center')

    s, x0, dx = 5.2, 17, 6.2
    rows = [('Carry-last', ORANGE, 27), ('InfiLoop', BLUE, 11)]
    for name, col, yb in rows:
        ax.text(3, yb + s / 2, name, fontsize=13.5, weight='bold', va='center')
        ax.add_patch(Rectangle((3, yb + s / 2 - 3.6), 9.5, .5, fc=col, ec='none'))
        infi = name == 'InfiLoop'
        prev = np.append(W[t - 1], 0) if t > 0 else np.zeros(1)
        wt = prev + (W[t] - prev) * a_upd
        for i in range(t + 1):
            x = x0 + i * dx
            if i == t:
                alpha = a_new
            elif infi:
                alpha = .3 + .7 * min(1, wt[i] / max(wt.max(), 1e-9))
            else:
                alpha = 1.0
            gone = not infi and i < t and (i < t - 1 or a_upd >= .5)
            grid(ax, x, yb, s, C[i], alpha=alpha, mono=FAINT if gone else None)
            if infi and wt[i] > 0:
                bh = 5.5 * wt[i]
                ax.add_patch(Rectangle((x, yb - 1.2 - bh), s * .9, bh, fc=BLUE, ec='none'))
        if infi:
            ax.text(x0 - 1.2, yb - 1.2, 'weight', fontsize=9.5, color=MUTED, ha='right', va='top')
        # arrow into state
        sx = 74
        ax.add_patch(FancyArrowPatch((x0 + (T - 1) * dx + s + .8, yb + s / 2), (sx - 1.5, yb + s / 2),
                                     arrowstyle='-|>', mutation_scale=16, lw=1.4, color=MUTED))
        # state
        ss = 8.5; sy = yb + s / 2 - ss / 2
        if infi:
            prev = H[t - 1] if t > 0 else np.zeros(16)
            h = prev + (H[t] - prev) * a_upd
            g = G[t]
            ax.text(sx + ss / 2, sy - 2.2, f'step size γ = {g:.2f}' if a_upd > 0 else ' ', fontsize=10.5,
                    color=INK, va='center', ha='center')
        else:
            prev = C[t - 1] if t > 0 else np.zeros(16)
            h = prev + (C[t] - prev) * a_upd
        grid(ax, sx, sy, ss, h, decode=True)
        ax.text(sx + ss / 2, sy + ss + 1.3, 'state', fontsize=10.5, color=MUTED, ha='center')
        solved = (h > .5).all()
        was = t >= 4 and a_upd >= 1
        if was:
            ax.text(sx + ss + 1.5, yb + s / 2, '✓ solved' if solved else '✗ lost', fontsize=13, weight='bold',
                    color=INK, va='center')
    fig.savefig(FR / f'{idx:04d}.png', facecolor=BG); plt.close(fig)

i = 0
for t in range(T):
    for k in range(5): frame(i, t, (k + 1) / 5, 0); i += 1
    for k in range(8): frame(i, t, 1, (k + 1) / 8); i += 1
    hold = 18 if t in (4, 5, 7) else 6
    for k in range(hold): frame(i, t, 1, 1); i += 1
for k in range(30): frame(i, T - 1, 1, 1); i += 1

gif = OUT / 'mechanism.gif'; pal = FR / 'pal.png'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '15', '-i', str(FR / '%04d.png'),
                '-vf', 'scale=900:-1:flags=lanczos,palettegen=max_colors=64:stats_mode=diff', str(pal)], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '15', '-i', str(FR / '%04d.png'), '-i', str(pal),
                '-lavfi', 'scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle',
                '-loop', '0', str(gif)], check=True)
print(gif, gif.stat().st_size // 1024, 'KiB', i, 'frames')
