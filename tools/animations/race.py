"""Animated test-time depth scaling on Sudoku-Extreme (paper Figure 3, left), from scaling.json."""
import json, os, subprocess, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('OUT', HERE / 'out'))
FR = HERE / 'frames_race'
shutil.rmtree(FR, ignore_errors=True); FR.mkdir(parents=True); OUT.mkdir(exist_ok=True)

data = json.loads((HERE / 'scaling.json').read_text())['scaling']
BLUE, PURPLE, ORANGE = '#3A5BD9', '#2BA89A', '#F08A4B'
INK, GRID, BG = '#1F2328', '#ECEEF1', '#FFFFFF'
MUTED = INK
SER = [('InfiLoop', BLUE, '-', 2.6), ('FPRM', PURPLE, '-', 1.8), ('TRM', ORANGE, (0, (5, 3)), 1.8)]
X0, X1 = 156, 24960

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.linewidth': .8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})

def interp(k, x):
    v = data[k]; return float(np.interp(np.log(x), np.log(v['layers']), v['accuracy']))

def frame(xcur, idx, callout=0.0, final=0.0):
    fig = plt.figure(figsize=(10, 5.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([.075, .14, .575, .66])
    fig.text(.075, .93, 'Same accuracy, half the layers', fontsize=18, weight='bold')
    fig.text(.075, .875, 'Sudoku-Extreme accuracy (%) vs. test-time layers',
             fontsize=11, color=MUTED)
    ax.set(xscale='log', xlim=(X0 * .92, X1 * 1.08), ylim=(50, 96))
    ax.set_xticks([156, 468, 1872, 7488, 24960], labels=['156', '468', '1,872', '7,488', '24,960'])
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_yticks([50, 60, 70, 80, 90])
    ax.grid(axis='y', color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.set_xlabel('Layers (log scale)', color=MUTED)
    ax.tick_params(length=3)
    ax.axvspan(xcur, X1 * 1.08, color='#F6F7F8', lw=0, zorder=0)
    for k, col, ls, lw in SER[::-1]:
        v = data[k]; L = np.array(v['layers']); A = np.array(v['accuracy'])
        m = L <= xcur
        xs = np.append(L[m], xcur); ys = np.append(A[m], interp(k, xcur))
        ax.plot(xs, ys, color=col, ls=ls, lw=lw, solid_capstyle='round', zorder=3)
        ax.scatter([xcur], [ys[-1]], s=70, color=col, edgecolor='white', linewidth=2, zorder=4)
    if callout > 0:
        a = min(callout, 1.0)
        yi, yf = interp('InfiLoop', 1872), interp('FPRM', 3744)
        ax.plot([1872, 1872], [50, yi], color=INK, ls=':', lw=1, alpha=a, zorder=2)
        ax.plot([3744, 3744], [50, yf], color=INK, ls=':', lw=1, alpha=a, zorder=2)
        ax.annotate('', xy=(1872, 66), xytext=(3744, 66), alpha=a,
                    arrowprops=dict(arrowstyle='<->', lw=1.2, color=INK, alpha=a))
        ax.text((1872 * 3744) ** .5, 64.5, 'half the layers', ha='center', va='top', fontsize=10.5,
                color=INK, alpha=a)
    # live readout panel
    px, py = .7, .74
    fig.text(px, py + .055, 'Layers', fontsize=10.5, color=MUTED)
    fig.text(px, py - .005, f'{int(round(xcur)):,}', fontsize=24, weight='bold', family='DejaVu Sans Mono')
    for j, (k, col, ls, lw) in enumerate(SER):
        y = py - .13 - j * .085
        fig.add_artist(plt.Line2D([px, px + .035], [y + .012, y + .012], color=col, ls=ls, lw=lw + .6,
                                  transform=fig.transFigure))
        fig.text(px + .045, y, k, fontsize=13, weight='bold' if k == 'InfiLoop' else 'normal')
        fig.text(px + .255, y, f'{interp(k, xcur):5.1f}%', fontsize=13, ha='right', family='DejaVu Sans Mono')
    fig.savefig(FR / f'{idx:04d}.png', facecolor=BG)
    plt.close(fig)

frames = []
xs1 = np.exp(np.linspace(np.log(X0), np.log(1872), 55))
xs2 = np.exp(np.linspace(np.log(1872), np.log(3744), 18))[1:]
xs3 = np.exp(np.linspace(np.log(3744), np.log(X1), 60))[1:]
for x in xs1: frames.append((x, 0, 0))
for t in range(24): frames.append((1872, (t + 1) / 8, 0))          # pause at 1,872, fade in callout
for x in xs2: frames.append((x, 1, 0))
for x in xs3: frames.append((x, 1, 0))
for t in range(50): frames.append((X1, 1, (t + 1) / 8))            # hold at the end
for i, (x, c, f) in enumerate(frames): frame(x, i, c, f)

gif = OUT / 'test_time_scaling.gif'
pal = FR / 'pal.png'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '20', '-i', str(FR / '%04d.png'),
                '-vf', 'scale=900:-1:flags=lanczos,palettegen=max_colors=64:stats_mode=diff', str(pal)], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '20', '-i', str(FR / '%04d.png'), '-i', str(pal),
                '-lavfi', 'scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle',
                '-loop', '0', str(gif)], check=True)
print(gif, gif.stat().st_size // 1024, 'KiB', len(frames), 'frames')
