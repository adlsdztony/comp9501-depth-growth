"""Static figures for the talk: search-vs-grow schematic, goal split, pilot insertion shock."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
DEPTH_COL = {1: '#86b6ef', 2: '#5598e7', 3: '#256abf', 4: '#0d366b'}
GROW_COL = '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def bar(ax, x0, x1, y, col, h=0.5, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=col, ec='none', alpha=alpha))


# 1. Search vs grow schematic
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(wspace=0.12))
for ax in (a, b):
    ax.set_xlim(-0.2, 3.3); ax.set_ylim(-0.6, 4.4); ax.axis('off')
for i, d in enumerate([4, 3, 2, 1]):
    bar(a, 0, 3, i, DEPTH_COL[d])
    a.text(-0.12, i, f'depth {d}', ha='right', va='center', color=INK2, fontsize=13)
a.text(1.5, 4.05, 'Search: K runs, keep the best', ha='center', color=INK, fontsize=16, weight='bold')
a.text(3.15, 1.5, '× K\ncost', ha='left', va='center', color=INK2, fontsize=13)
for (x0, x1, d) in [(0, 1, 1), (1, 2, 2), (2, 3, 3)]:
    bar(b, x0 + (0.02 if x0 else 0), x1, 1.5, DEPTH_COL[d] if False else GROW_COL, alpha=0.45 + 0.25 * (d - 1))
    b.text((x0 + x1) / 2, 1.5, f'depth {d}', ha='center', va='center', color='white', fontsize=13, weight='bold')
for x in (1, 2):
    b.annotate('', xy=(x, 2.05), xytext=(x, 2.75), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6))
    b.text(x, 2.95, '+ layer', ha='center', color=INK, fontsize=12)
b.text(1.5, 4.05, 'Grow: one run, add depth on the way', ha='center', color=INK, fontsize=16, weight='bold')
b.text(3.15, 1.5, '× 1\ncost', ha='left', va='center', color=INK2, fontsize=13)
fig.savefig(OUT / 'schematic_search_vs_grow.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# 2. Goal split (per task, 50 goal variants)
fig, ax = plt.subplots(figsize=(12, 1.9))
ax.set_xlim(0, 50); ax.set_ylim(0, 1); ax.axis('off')
for i in range(50):
    c = '#b7d3f6' if i < 30 else ('#9085e9' if i < 40 else GROW_COL)
    ax.add_patch(FancyBboxPatch((i + 0.08, 0.35), 0.84, 0.5, boxstyle='round,pad=0,rounding_size=0.12', fc=c, ec='none'))
for x, t in [(15, 'train goals (30)'), (35, 'validation (10)'), (45, 'held-out test (10)')]:
    ax.text(x, 0.08, t, ha='center', color=INK, fontsize=14)
fig.savefig(OUT / 'goal_split.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# 3. Pilot: insertion shock (from evidence/summary.csv; one seed, 3 tasks, 60 episodes)
arms = ['Keep', 'Identity', 'Random']
s0, s1, kl = [33, 33, 0], [40, 40, 20], [0.0, 0.0, 1.213]
cols = ['#86b6ef', GROW_COL, '#52514e']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.35))
ax = axes[0]
x = np.arange(3)
ax.bar(x - 0.2, np.array(s0) / 60 * 100, 0.36, color=cols, alpha=0.45, edgecolor='none')
ax.bar(x + 0.2, np.array(s1) / 60 * 100, 0.36, color=cols, edgecolor='none')
for i in range(3):
    ax.text(i - 0.2, s0[i] / 60 * 100 + 2, f'{s0[i] / 60 * 100:.0f}', ha='center', fontsize=12, color=INK2)
    ax.text(i + 0.2, s1[i] / 60 * 100 + 2, f'{s1[i] / 60 * 100:.0f}', ha='center', fontsize=12, color=INK)
ax.set_xticks(x, arms); ax.set_ylim(0, 100); ax.set_ylabel('Success (%)')
ax.set_title('light: at insert   dark: +300k steps', fontsize=13, color=INK2)
ax2 = axes[1]
ax2.bar(x, kl, 0.55, color=cols, edgecolor='none')
for i in range(3):
    ax2.text(i, kl[i] + 0.04, f'{kl[i]:.2f}', ha='center', fontsize=12)
ax2.set_xticks(x, ['Keep', 'Identity', 'Random']); ax2.set_ylabel('KL(parent ‖ new) at insert')
ax2.set_ylim(0, 1.45)
for a_ in axes:
    a_.spines[['top', 'right']].set_visible(False); a_.grid(axis='y', color=GRID); a_.set_axisbelow(True)
fig.savefig(OUT / 'pilot_insertion.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('ok')
