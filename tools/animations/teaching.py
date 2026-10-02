import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'teaching.json').read_text())
X = np.asarray(DATA['outer_updates'])
TRM = np.asarray(DATA['series']['TRM'])
OURS = np.asarray(DATA['series']['InfiLoop'])
PEAK = float(TRM.max())
BG, INK, MUTED = '#FFFFFF', '#24292F', '#63707D'
ORANGE, BLUE, GOLD = '#C96521', '#1768AE', '#9A6500'
FPS, DURATION = 24, 18

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'text.color': INK, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .8, 'axes.edgecolor': '#9AA4AF',
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
})


def smooth(v):
    v = float(np.clip(v, 0, 1))
    return v*v*(3-2*v)


def fade(t, start, end, duration=.35):
    return smooth((t-start)/duration) * (1-smooth((t-end)/duration))


def point_trace(ax, values, end, color, opacity=1):
    xx = np.r_[X[X < end], end]
    yy = np.interp(xx, X, values)
    ax.plot(xx, yy, color=color, lw=2.7, alpha=opacity,
            solid_capstyle='round', zorder=5)
    ax.scatter([end], [yy[-1]], s=42, color=color, linewidth=0,
               alpha=opacity, zorder=8)
    return yy[-1]


def render(t):
    z = smooth((t-5.0)/1.25) * (1-smooth((t-11.1)/1.15))
    if t < 1:
        end = 1.
    elif t < 4:
        end = 1 + 17 * smooth((t-1)/3)
    elif t < 6.3:
        end = 18.
    elif t < 9.7:
        end = 18 + 54 * ((t-6.3)/3.4)
    else:
        end = 72.
    blue_end = 1 + 71*smooth((t-12.6)/2.8)
    trm_alpha = 1-.30*smooth((t-12.4)/.6)
    final_alpha = smooth((t-15.4)/.5)
    trm_alpha += .22*final_alpha
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([.108, .255, .748, .466], facecolor=BG)
    ax.set_xticks([18, 36, 54, 72] if z > .45 else [0, 18, 36, 54, 72])
    ax.set_yticks([54, 58, 62, 66] if z > .45 else [0, 25, 50, 75, 100])
    ax.set(xlim=(12*z, 82), ylim=(54*z, 100-34*z))
    ax.tick_params(length=4, width=.8, pad=9, labelsize=11)
    ax.set_xlabel('Outer updates', labelpad=12, fontsize=12)
    ax.grid(axis='y', color='#D8DEE5', alpha=.65, linewidth=.5)
    ax.set_axisbelow(True)

    chapters = [
        (0, 4.65, '01   LET THE MODEL KEEP THINKING', 'At first, more loops help.'),
        (4.8, 10.85, '02   LOOK PAST THE PEAK', 'Then, accuracy starts to fall.'),
        (11.15, 18, '03   COMPARE THE SAME UPDATE BUDGET', 'InfiLoop keeps improving.'),
    ]
    for start, end_title, kicker, title in chapters:
        a = fade(t, start-.35 if start == 0 else start, end_title)
        fig.text(.5, .927, kicker, fontsize=9.5, color=MUTED, ha='center', alpha=a)
        fig.text(.5, .832, title, fontsize=30, family='STIXGeneral', ha='center', alpha=a)
    fig.text(.108, .753, 'Exact accuracy (%)', fontsize=11, color=MUTED)
    zoom_label = 'Magnified · 54–66%'
    if 5 < t < 6.25:
        zoom_label = 'Zooming in'
    elif 11.1 < t < 12.25:
        zoom_label = 'Returning to full scale'
    fig.text(.856, .753, zoom_label, fontsize=10.5, color=GOLD,
             ha='right', alpha=z)

    y = point_trace(ax, TRM, end, ORANGE, trm_alpha)
    if end > 5:
        a = (1-smooth((t-3.85)/.3)) * smooth((end-5)/3)
        ax.annotate('TRM', (end, y), xytext=(11, -6), textcoords='offset points',
                    color=ORANGE, fontsize=16, family='STIXGeneral', alpha=a)

    peak_alpha = smooth((t-4.0)/.3)
    peak_text_alpha = peak_alpha*trm_alpha*(1-smooth((t-12.35)/.3)+smooth((t-13.8)/.35))
    if peak_alpha:
        ax.scatter([18], [PEAK], s=49, facecolor=BG, edgecolor=GOLD,
                   linewidth=1.6, alpha=peak_alpha*trm_alpha, zorder=9)
        # A single expanding ring draws attention at the moment of the peak.
        pulse = np.clip((t-4.0)/.8, 0, 1)
        if 0 < pulse < 1:
            ax.scatter([18], [PEAK], s=50+1000*pulse, facecolor='none',
                       edgecolor=GOLD, linewidth=1.2, alpha=1-pulse, zorder=9)
        guide_end = 18+(max(18,end)-18)
        ax.plot([18, guide_end], [PEAK, PEAK], color=GOLD, lw=.95,
                ls=(0,(4,5)), alpha=.55*peak_alpha*trm_alpha, zorder=3)
        ax.annotate('62.2%', (18,PEAK), xytext=(0,22), textcoords='offset points',
                    ha='center', color=GOLD, fontsize=21, family='STIXGeneral',
                    alpha=peak_text_alpha)
        ax.annotate('peak at update 18', (18,PEAK), xytext=(0,8),
                    textcoords='offset points', ha='center', color=MUTED,
                    fontsize=8.5, alpha=peak_text_alpha)

    if t > 6.3:
        measured = min(72, int(np.floor(end + 1e-8)))
        xx = np.r_[X[(X >= 18)&(X < end)],end]
        yy = np.interp(xx,X,TRM)
        ax.fill_between(xx, yy, PEAK, color=ORANGE, alpha=.07*trm_alpha, linewidth=0)
        a = smooth((t-6.3)/.35)*(1-smooth((t-11.1)/.5))
        ax.annotate(f'{TRM[measured-1]:.1f}%', (end,y), xytext=(11,-5),
                    textcoords='offset points', color=ORANGE, fontsize=21,
                    family='STIXGeneral', alpha=a)
        if t > 9.8:
            a = fade(t, 9.8,11.05)
            ax.annotate('', (68,TRM[-1]), (68,PEAK),
                        arrowprops=dict(arrowstyle='<->',color=INK,lw=1.2,alpha=a))
            ax.text(64.5,(PEAK+TRM[-1])/2,'−4.1 pp',ha='right',va='center',
                    color=INK,fontsize=18,family='STIXGeneral',alpha=a)

    if t >= 12.6:
        a=smooth((t-12.6)/.3)
        by=point_trace(ax,OURS,blue_end,BLUE,a)
        if blue_end > 7:
            label_shift = smooth((blue_end-18)/8)
            ax.annotate('InfiLoop', (blue_end,by),xytext=(10*label_shift,26-14*label_shift),
                        textcoords='offset points',color=BLUE,fontsize=18,
                        family='STIXGeneral',alpha=a)
        if final_alpha:
            ax.annotate('90.9%',(72,OURS[-1]),xytext=(10,-10),textcoords='offset points',
                        color=BLUE,fontsize=19,family='STIXGeneral',alpha=final_alpha)
    if t > 11.9:
        a=smooth((t-11.9)/.5)
        ax.annotate('TRM',(72,TRM[-1]),xytext=(10,-5),textcoords='offset points',
                    color=ORANGE,fontsize=18,family='STIXGeneral',alpha=a)
        ax.annotate('58.1%',(72,TRM[-1]),xytext=(10,-25),textcoords='offset points',
                    color=ORANGE,fontsize=19,family='STIXGeneral',alpha=a)

    captions=[
        (0,3.9,'Run the same checkpoint for more outer updates.',INK),
        (4.0,5.8,'18 updates · 468 executed layers',GOLD),
        (6.0,9.4,'Keep looping beyond the best answer rate.',INK),
        (9.55,11.05,'At update 72: 58.1%, down from 62.2%.',ORANGE),
        (11.55,15.1,'Now follow InfiLoop over the same 72 updates.',BLUE),
        (15.3,18,'TRM loses 4.1 points from its peak. InfiLoop reaches 90.9%.',INK),
    ]
    for start,finish,caption,color in captions:
        a=fade(t,start-.35 if start==0 else start,finish)
        fig.text(.5,.125,caption,ha='center',fontsize=14,color=color,alpha=a)
    fig.text(.5,.044,'Sudoku-Extreme · Figure 1(a) · 32,768 puzzles · TRM self-attention checkpoint',
             color=MUTED,fontsize=8.3,ha='center')
    # The focus change is a camera zoom; source coordinates never change.
    return fig


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stills',action='store_true')
    ap.add_argument('--out',type=Path,default=HERE.parents[1]/'previews/loop_accuracy_white')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for name,t in [('rise',3.2),('peak',4.55),('zoom',6.3),('fall',8.2),('drop',10.5),('comparison',14.0),('final',17)]:
        fig=render(t);fig.savefig(args.out/f'{name}.png',facecolor=BG)
        if name=='final':fig.savefig(args.out/'final.svg',facecolor=BG)
        plt.close(fig)
    if args.stills:return
    video=args.out/'loop_accuracy.mp4';gif=args.out/'loop_accuracy.gif'
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
                              '-vcodec','rawvideo','-pix_fmt','rgba','-s','1280x720',
                              '-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','17',
                              '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
                             stdin=subprocess.PIPE)
    try:
        for i in range(FPS*DURATION):
            fig=render(i/FPS)
            fig.canvas.draw()
            encoder.stdin.write(fig.canvas.buffer_rgba())
            plt.close(fig)
    finally:
        encoder.stdin.close()
        code=encoder.wait()
    if code:
        raise RuntimeError(f'Video encoding failed: {code}')
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex',
                    'fps=24,scale=1120:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];'
                    '[b][p]paletteuse=dither=none:diff_mode=rectangle','-loop','0',str(gif)],check=True)
    source={**DATA,'style_references':['https://www.3blue1brown.com/lessons/neural-networks/',
            'https://docs.manim.community/en/stable/examples.html'],
            'presentation':'TRM trace, labelled camera zoom to 54–66%, return to full axes, InfiLoop trace at the same 72-update budget.',
            'seconds':DURATION,'fps':FPS}
    (args.out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    print(video);print(gif)


if __name__=='__main__':main()
