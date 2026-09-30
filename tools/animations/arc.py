"""Animated ARC-AGI-2 comparison (paper Figure 6), from arc2.json: real predictions, best of two attempts."""
import json, os, subprocess, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('OUT', HERE / 'out'))
FR = HERE / 'frames_arc'
shutil.rmtree(FR, ignore_errors=True); FR.mkdir(parents=True); OUT.mkdir(exist_ok=True)

cases = json.loads((HERE / 'arc2.json').read_text())['cases']
ARC = ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                      '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
BLUE, ORANGE = '#3A5BD9', '#F08A4B'
INK, BG = '#1F2328', '#FFFFFF'
MUTED = INK
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})
FH = 4.4

def draw_grid(fig, rect, g, alpha=1.0, wrong=None, pulse=0.0):
    ax = fig.add_axes(rect); g = np.array(g); h, w = g.shape
    dim = 1 - .72 * pulse if wrong is not None and wrong.any() else 1.0
    ax.pcolormesh(g[::-1], cmap=ARC, vmin=0, vmax=9, edgecolors='#3A3F45', linewidth=.25, alpha=alpha * dim)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect('equal', anchor='N'); ax.axis('off')
    if wrong is not None and pulse > 0:
        for (r, c) in zip(*np.nonzero(wrong)):
            y = h - 1 - r
            ax.add_patch(Rectangle((c, y), 1, 1, fc=ARC(int(g[r, c])), ec='none', alpha=alpha))
            ax.add_patch(Rectangle((c, y), 1, 1, fill=False, ec='black', lw=3.4, alpha=pulse))
            ax.add_patch(Rectangle((c, y), 1, 1, fill=False, ec='white', lw=1.7, alpha=pulse))
    return ax

def frame(idx, case, t_in, t_pred, pulse, n_shown):
    fig = plt.figure(figsize=(10, FH), dpi=100, facecolor=BG)
    fig.text(.04, .915, 'ARC-AGI-2', fontsize=18, weight='bold')
    fig.text(.04, .845, f"task {case['task']}",
             fontsize=11, color=MUTED)
    tgt = np.array(case['target'])
    preds = {k: np.array(case['predictions'][k]['grid']) for k in ('InfiLoop', 'TRM')}
    top, H = .72, .62
    cols = [('Demonstration', .03, .135, case['train']['input']), ('', .175, .135, case['train']['output']),
            ('Test', .345, .19, case['test_input'])]
    for title, x, wdt, g in cols:
        draw_grid(fig, [x, top - H, wdt, H], g, alpha=t_in)
    fig.text(.03, top + .02, 'Example', fontsize=11.5, color=MUTED, alpha=t_in)
    dh, dw = np.array(case['train']['input']).shape
    fig.text(.1675, top - 1.35 * dh / dw / FH / 2, '→', fontsize=15, ha='center', va='center', color=MUTED, alpha=t_in)
    fig.text(.345, top + .02, 'Test', fontsize=11.5, color=MUTED, alpha=t_in)
    fig.add_artist(plt.Line2D([.555, .555], [.1, .78], color='#D5D9DD', lw=1, transform=fig.transFigure))
    for j, (k, col) in enumerate([('InfiLoop', BLUE), ('TRM', ORANGE)]):
        x = .58 + j * .21
        wrong = preds[k] != tgt
        draw_grid(fig, [x, top - H, .19, H], preds[k], alpha=t_pred, wrong=wrong, pulse=pulse)
        fig.text(x, top + .02, k, fontsize=13, weight='bold', color=INK, alpha=t_pred)
        fig.add_artist(plt.Line2D([x, x + .19], [top + .005, top + .005], color=col, lw=3,
                                  transform=fig.transFigure, alpha=t_pred))
        if pulse > 0:
            nw = int(wrong.sum())
            msg = '✓ correct' if nw == 0 else f'✗ {min(n_shown, nw)} wrong'
            fig.text(x, top - 1.9 * tgt.shape[0] / tgt.shape[1] / FH - .1, msg, fontsize=13, weight='bold', color=INK, alpha=min(1, pulse * 2))
    fig.savefig(FR / f'{idx:04d}.png', facecolor=BG); plt.close(fig)

i = 0
for case in cases:
    nw = int((np.array(case['predictions']['TRM']['grid']) != np.array(case['target'])).sum())
    for t in range(10): frame(i, case, (t + 1) / 10, 0, 0, 0); i += 1
    for t in range(10): frame(i, case, 1, (t + 1) / 10, 0, 0); i += 1
    for t in range(8): frame(i, case, 1, 1, 0, 0); i += 1
    for t in range(16): frame(i, case, 1, 1, min(1, (t + 1) / 6), round(nw * (t + 1) / 16)); i += 1
    for t in range(30):
        frame(i, case, 1, 1, .55 + .45 * np.cos(t / 30 * 3 * np.pi) ** 2, nw); i += 1
    for t in range(8): frame(i, case, 1 - (t + 1) / 8, 1 - (t + 1) / 8, 0, nw); i += 1

gif = OUT / 'arc_agi_2.gif'; pal = FR / 'pal.png'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '15', '-i', str(FR / '%04d.png'),
                '-vf', 'scale=900:-1:flags=lanczos,palettegen=max_colors=96:stats_mode=diff', str(pal)], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '15', '-i', str(FR / '%04d.png'), '-i', str(pal),
                '-lavfi', 'scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle',
                '-loop', '0', str(gif)], check=True)
print(gif, gif.stat().st_size // 1024, 'KiB', i, 'frames')
