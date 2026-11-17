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

# 4. Identity insertion schematic
def block(ax, x, y, w, h, text, fc, tc='white', sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec='none'))
    ax.text(x + w / 2, y + h / 2 + (0.09 if sub else 0), text, ha='center', va='center', color=tc, fontsize=14, weight='bold')
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.17, sub, ha='center', va='center', color=tc, fontsize=11)


def arrow(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.6))


fig, ax = plt.subplots(figsize=(14, 5.0))
ax.set_xlim(0, 14.2); ax.set_ylim(-0.5, 4.5); ax.axis('off')
KEEP = '#256abf'; HEAD = '#52514e'
for y, new in [(2.9, False), (0.8, True)]:
    ax.text(0.0, y + 1.0, 'After inserting a layer' if new else 'Before', fontsize=15, weight='bold', color=INK)
    ax.text(0.3, y + 0.35, 's', fontsize=18, style='italic', ha='center', va='center', color=INK)
    arrow(ax, 0.6, 1.1, y + 0.35)
    block(ax, 1.1, y, 2.5, 0.7, 'Linear + ReLU', KEEP, sub='trained layers')
    arrow(ax, 3.7, 4.25, y + 0.35)
    ax.text(4.8, y + 0.35, 'h ≥ 0', fontsize=16, ha='center', va='center', color=INK)
    if new:
        arrow(ax, 5.35, 5.9, y + 0.35)
        block(ax, 5.9, y, 3.3, 0.7, 'Identity layer + ReLU', GROW_COL, sub='W = I, no bias')
        arrow(ax, 9.3, 9.8, y + 0.35)
        ax.text(10.05, y + 0.35, 'h', fontsize=16, ha='center', va='center', color=INK)
        arrow(ax, 10.3, 10.75, y + 0.35)
    else:
        arrow(ax, 5.35, 10.75, y + 0.35)
    block(ax, 10.75, y, 2.1, 0.7, 'Policy head', HEAD, sub='mean, std')
    arrow(ax, 12.95, 13.35, y + 0.35)
    ax.text(13.8, y + 0.35, 'π(a|s)', fontsize=16, ha='center', va='center', color=INK)
ax.text(7.1, -0.25, 'h ≥ 0  ⇒  ReLU(I·h) = h  ⇒  the policy is identical at the moment of insertion',
        ha='center', va='center', fontsize=15, color=INK)
fig.savefig(OUT / 'identity_schematic.png', dpi=200, bbox_inches='tight', pad_inches=0.1); plt.close(fig)
print('schematic ok')

# 5. Benchmark: MT10 task grid
tasks = ['reach', 'push', 'pick-place', 'door-open', 'drawer-open',
         'drawer-close', 'button-press\ntopdown', 'peg-insert-side', 'window-open', 'window-close']
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis('off')
for i, t in enumerate(tasks):
    r, c = divmod(i, 5)
    x, y = c * 2.0 + 0.08, 2.3 - r * 2.0
    ax.add_patch(FancyBboxPatch((x, y), 1.84, 1.6, boxstyle='round,pad=0,rounding_size=0.1', fc='#f1f4f9', ec='#c9d6ea', lw=1.2))
    ax.text(x + 0.92, y + 1.05, f'{i + 1}', ha='center', va='center', fontsize=22, color='#256abf', weight='bold')
    ax.text(x + 0.92, y + 0.45, t, ha='center', va='center', fontsize=12.5, color=INK)
fig.savefig(OUT / 'mt10_tasks.png', dpi=200, bbox_inches='tight', pad_inches=0.05); plt.close(fig)

# 6. Learner: multi-task PPO loop
fig, ax = plt.subplots(figsize=(13, 5.0))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.0); ax.axis('off')
def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec='none'))
    ax.text(x + w / 2, y + h * 0.64, title, ha='center', va='center', color='white', fontsize=14, weight='bold')
    ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', color='white', fontsize=11)
def arr(x0, y0, x1, y1, rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.8, connectionstyle=f'arc3,rad={rad}'))
NAV, BLU, GRY = '#14213d', '#256abf', '#52514e'
box(0.2, 3.0, 3.0, 1.3, '10 environments', 'one per MT10 task', GRY)
box(4.0, 3.0, 2.6, 1.3, 'Rollout', '100k steps per iteration', BLU)
box(7.4, 3.0, 2.4, 1.3, 'Advantages', 'GAE, γ 0.99, λ 0.97', BLU)
box(10.6, 3.0, 2.3, 1.3, 'PPO update', '16 epochs, clip 0.2', NAV)
arr(3.25, 3.65, 3.95, 3.65); arr(6.65, 3.65, 7.35, 3.65); arr(9.85, 3.65, 10.55, 3.65)
box(2.2, 0.4, 4.2, 1.5, 'Actor (policy)', 'MLP width 400, depth 1–4  ← grown', GROW_COL)
box(7.0, 0.4, 4.2, 1.5, 'Critic (value)', '3 × 400, LayerNorm, PopArt per task', GRY)
arr(11.75, 2.95, 9.2, 1.95, rad=-0.15)
arr(11.75, 2.95, 5.0, 1.95, rad=-0.05)
arr(3.6, 1.95, 1.8, 2.95, rad=-0.2)
ax.text(0.95, 2.35, 'acts in', fontsize=11, color=INK2)
ax.text(6.5, 2.45, 'gradient steps on both networks', fontsize=11, color=INK2, ha='center')
fig.savefig(OUT / 'ppo_learner.png', dpi=200, bbox_inches='tight', pad_inches=0.05); plt.close(fig)
print('benchmark/learner ok')

# 7. Related-work comparison matrix
rows = [('Grid / random search', 'any'),
        ('Successive halving, Hyperband', 'any'),
        ('Population-based training', 'RL, supervised'),
        ('Large regularised network (BRO, SimBa)', 'RL'),
        ('Network growth (Net2Net, GradMax)', 'supervised'),
        ('Ours: identity growth in PPO', 'multi-task RL')]
cols = ['One training\nrun', 'Starts\nsmall', 'Grows during\ntraining', 'Keeps policy\nat insertion', 'Studied in']
M = [['no', 'no', 'no', '–'], ['no', 'no', 'no', '–'], ['no', 'no', 'no', '–'],
     ['yes', 'no', 'no', '–'], ['yes', 'yes', 'yes', 'yes'], ['yes', 'yes', 'yes', 'yes']]
fig, ax = plt.subplots(figsize=(14.5, 5.6))
ax.set_xlim(0, 14.2); ax.set_ylim(0, len(rows) + 1.2); ax.axis('off')
X0, CW = 5.1, 1.82
for j, c in enumerate(cols):
    ax.text(X0 + j * CW + CW / 2, len(rows) + 0.55, c, ha='center', va='center', fontsize=12, color=INK, weight='bold')
for i, (name, setting) in enumerate(rows):
    y = len(rows) - 1 - i
    ours = name.startswith('Ours')
    if ours:
        ax.add_patch(FancyBboxPatch((0.05, y + 0.08), 14.1, 0.84, boxstyle='round,pad=0,rounding_size=0.08',
                                    fc='#fde8df', ec=GROW_COL, lw=1.6))
    elif i % 2 == 0:
        ax.add_patch(plt.Rectangle((0.05, y + 0.08), 14.1, 0.84, fc='#f4f6fa', ec='none'))
    ax.text(0.25, y + 0.5, name, va='center', fontsize=13, color=INK, weight='bold' if ours else 'normal')
    for j, v in enumerate(M[i]):
        cx = X0 + j * CW + CW / 2
        if v == 'yes':
            ax.scatter([cx], [y + 0.5], s=620, color=GROW_COL if ours else '#256abf', zorder=3)
            ax.text(cx, y + 0.5, '✓', ha='center', va='center', fontsize=13, color='white', weight='bold', zorder=4)
        elif v == 'no':
            ax.scatter([cx], [y + 0.5], s=620, facecolor='white', edgecolor='#b9b8b3', linewidth=1.4, zorder=3)
        else:
            ax.text(cx, y + 0.5, 'n/a', ha='center', va='center', fontsize=11, color='#9a9994')
    ax.text(X0 + 4 * CW + CW / 2, y + 0.5, setting, ha='center', va='center', fontsize=12,
            color=GROW_COL if ours else INK2, weight='bold' if ours else 'normal')
fig.savefig(OUT / 'related_matrix.png', dpi=200, bbox_inches='tight', pad_inches=0.05); plt.close(fig)
print('matrix ok')
