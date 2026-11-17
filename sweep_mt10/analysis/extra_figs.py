"""Backup analysis figures (main-result robustness + earlier findings).

usage: python extra_figs.py RUNS_DIR FIG_DIR
Reuses load/flops/cost helpers from aggregate.py; writes to FIG_DIR/backup/.
"""
import itertools, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_src = (HERE / 'aggregate.py').read_text(encoding='utf8').replace("if __name__ == '__main__':\n    main()", '')
G = {'__name__': 'aggregate_lib'}
exec(compile(_src, 'aggregate.py', 'exec'), G)
plt, load, flops, cum_flops, at, color = G['plt'], G['load'], G['flops'], G['cum_flops'], G['at'], G['color']
INK, INK2, GRID, SURF = G['INK'], G['INK2'], G['GRID'], G['SURF']
DEPTH_COL, GROW_COL, GROWE_COL = G['DEPTH_COL'], G['GROW_COL'], G['GROWE_COL']

OUT = Path(sys.argv[2]) / 'backup'; OUT.mkdir(parents=True, exist_ok=True)
T = 3_000_000
STEPS = [i * 500_000 for i in range(7)]
FIXED = ['d1', 'd2', 'd3', 'd4']
SC = {'Best depth': '#0d366b', 'Grid search': '#52514e', 'Successive halving': '#008300', 'Grow early': GROWE_COL,
      'Grow late': GROW_COL, 'Default depth 2': '#5598e7', 'Random depth': '#86b6ef', 'Depth 4': '#0d366b'}
TASKS_SHORT = lambda t: t.replace('-v3', '').replace('-v2', '')


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight', pad_inches=0.15); plt.close(fig)
    print('wrote', name)


runs_all = [r for r in load() if r['done']]
runs = [r for r in runs_all if r['width'] == 400]
seeds = sorted({r['seed'] for r in runs if all(any(x['seed'] == r['seed'] and x['key'] == k for x in runs) for k in FIXED)})
BY = {s: {r['key']: r for r in runs if r['seed'] == s} for s in seeds}
dev = lambda by, k, t: at(by[k], t, 'success_dev') or 0
tst = lambda by, k, t: at(by[k], t, 'success_test') or 0
CF = {s: {k: [cum_flops(BY[s][k], t) for t in STEPS] for k in BY[s]} for s in seeds}
IDX = {t: i for i, t in enumerate(STEPS)}


def sim_sh(s, r1, keep1, r2=None, cands=FIXED):
    """Successive halving on logged curves: all cands to r1, keep `keep1` by validation, then (if keep1>1) to r2 and keep 1."""
    by, F = BY[s], CF[s]
    alive = sorted(cands, key=lambda k: -dev(by, k, r1))[:keep1]
    cost = sum(F[k][IDX[r1]] for k in cands)
    if keep1 == 1 or r2 is None:
        win = alive[0]
        cost += F[win][-1] - F[win][IDX[r1]]
    else:
        cost += sum(F[k][IDX[r2]] - F[k][IDX[r1]] for k in alive)
        win = max(alive, key=lambda k: dev(by, k, r2))
        cost += F[win][-1] - F[win][IDX[r2]]
    return cost, tst(by, win, T), win


def strategies(s):
    by, F = BY[s], CF[s]
    best = max(FIXED, key=lambda k: dev(by, k, T))
    orc = max(FIXED, key=lambda k: tst(by, k, T))
    shc, sht, shw = sim_sh(s, 500_000, 2, 1_500_000)
    out = {'Best depth': (F[orc][-1], tst(by, orc, T), orc),
           'Grid search': (sum(F[k][-1] for k in FIXED), tst(by, best, T), best),
           'Successive halving': (shc, sht, shw),
           'Grow early': (F['growE'][-1], tst(by, 'growE', T), 'growE'),
           'Grow late': (F['grow'][-1], tst(by, 'grow', T), 'grow'),
           'Default depth 2': (F['d2'][-1], tst(by, 'd2', T), 'd2'),
           'Random depth': (np.mean([F[k][-1] for k in FIXED]), np.mean([tst(by, k, T) for k in FIXED]), 'avg')}
    return out


ST = {s: strategies(s) for s in seeds}
NAMES = ['Best depth', 'Grid search', 'Successive halving', 'Grow early', 'Grow late', 'Default depth 2', 'Random depth']
n = len(seeds)
tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}[n]


def style_axes(ax, grid_axis='y'):
    ax.grid(axis='x' if grid_axis == 'y' else 'y', visible=False)


# ---------- M1 paired differences ----------
def m1():
    comps = ['Best depth', 'Successive halving', 'Default depth 2', 'Random depth', 'Grow late']
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, c in enumerate(comps):
        d = np.array([(ST[s]['Grow early'][1] - ST[s][c][1]) * 100 for s in seeds])
        y = len(comps) - 1 - i
        ci = tcrit * d.std(ddof=1) / np.sqrt(n)
        ax.plot([d.mean() - ci, d.mean() + ci], [y, y], color=SC[c], lw=2.5)
        ax.scatter(d, [y + 0.18] * n, s=22, color=SC[c], alpha=0.55, zorder=3)
        ax.scatter([d.mean()], [y], s=110, color=SC[c], edgecolor=SURF, linewidth=1.5, zorder=4)
        ax.text(14.5, y, f'{d.mean():+.1f} ± {ci:.1f}', va='center', fontsize=12, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(comps)), [f'vs {c.lower()}' for c in comps][::-1])
    ax.set_xlabel(f'Grow early minus baseline, held-out success (points)   mean ± 95% CI, small dots = seeds (n={n})')
    ax.set_xlim(-16, 20); style_axes(ax); ax.tick_params(axis='y', length=0)
    save(fig, 'm1_paired_differences.png')


# ---------- M2 per-seed results ----------
def m2():
    show = ['Best depth', 'Successive halving', 'Grow early', 'Default depth 2', 'Random depth', 'Grow late']
    marks = {'Best depth': '*', 'Successive halving': 'D', 'Grow early': 'o', 'Default depth 2': 's', 'Random depth': 'v', 'Grow late': 'o'}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for j, nm in enumerate(show):
        off = (j - (len(show) - 1) / 2) * 0.11
        ys = [ST[s][nm][1] * 100 for s in seeds]
        ax.scatter([i + off for i in range(n)], ys, s=150 if marks[nm] == '*' else 70, marker=marks[nm], color=SC[nm],
                   edgecolor=SURF, linewidth=1.0, zorder=3, label=nm)
    ax.set_xticks(range(n), [f'Seed {s}' for s in seeds]); ax.set_ylabel('Held-out success (%)')
    ax.set_ylim(70, 95); style_axes(ax)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=11)
    save(fig, 'm2_per_seed.png')


# ---------- M3 ranks ----------
def m3():
    show = ['Grid search', 'Successive halving', 'Grow early', 'Default depth 2', 'Random depth', 'Grow late']
    R = np.zeros((len(show), n))
    for j, s in enumerate(seeds):
        vals = np.array([ST[s][nm][1] for nm in show])
        order = (-vals).argsort().argsort().astype(float) + 1
        for v in np.unique(vals):  # average ties
            m = vals == v; order[m] = order[m].mean()
        R[:, j] = order
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('r', ['#0d366b', '#5598e7', '#cde2fb', '#f4f8fd'])
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.imshow(R, cmap=cmap, vmin=1, vmax=len(show), aspect='auto')
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f'{R[i, j]:g}', ha='center', va='center', fontsize=12, color='white' if R[i, j] <= 2.5 else INK)
    ax.set_xticks(range(n + 0), [f'Seed {s}' for s in seeds])
    ax.set_yticks(range(len(show)), [f'{nm}  (mean {R[i].mean():.1f})' for i, nm in enumerate(show)])
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title('Rank by held-out success within each seed (1 = best; ties averaged)', fontsize=13, color=INK2)
    save(fig, 'm3_ranks.png')


# ---------- M4 regret ----------
def m4():
    show = ['Grid search', 'Successive halving', 'Grow early', 'Default depth 2', 'Random depth', 'Grow late']
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, nm in enumerate(show):
        r = np.array([(ST[s]['Best depth'][1] - ST[s][nm][1]) * 100 for s in seeds])
        ax.bar(i, r.mean(), 0.6, color=SC[nm], edgecolor='none')
        ax.scatter([i] * n, r, s=20, color=INK, zorder=3)
        ax.text(i + 0.33, r.mean(), f'{r.mean():.1f}', ha='left', va='center', fontsize=12)
    ax.set_xticks(range(len(show)), [s.replace(' ', '\n', 1) for s in show])
    ax.set_ylabel('Gap to best depth (points)'); style_axes(ax)
    save(fig, 'm4_regret.png')


# ---------- M5 successive-halving variants ----------
def m5():
    pts = []
    r1s = [500_000, 1_000_000, 1_500_000, 2_000_000]
    for r1 in r1s:
        for keep1 in (1, 2, 3):
            r2s = [None] if keep1 == 1 else [r for r in STEPS if r1 < r < T]
            for r2 in r2s:
                c = np.mean([sim_sh(s, r1, keep1, r2)[0] for s in seeds]) / 1e15
                v = np.mean([sim_sh(s, r1, keep1, r2)[1] for s in seeds]) * 100
                pts.append((c, v, r1, keep1, r2))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    P = np.array([(p[0], p[1]) for p in pts])
    ax.scatter(P[:, 0], P[:, 1], s=40, color='#008300', alpha=0.35, edgecolor='none', label=f'Successive-halving variants ({len(pts)})')
    d = [p for p in pts if p[2] == 500_000 and p[3] == 2 and p[4] == 1_500_000][0]
    ax.scatter([d[0]], [d[1]], s=140, marker='D', color='#008300', edgecolor=SURF, zorder=4, label='Variant used in the main result')
    ge = np.mean([ST[s]['Grow early'][0] for s in seeds]) / 1e15, np.mean([ST[s]['Grow early'][1] for s in seeds]) * 100
    gr = np.mean([ST[s]['Grid search'][0] for s in seeds]) / 1e15, np.mean([ST[s]['Grid search'][1] for s in seeds]) * 100
    ax.scatter(*ge, s=170, color=GROWE_COL, edgecolor=SURF, zorder=5, label='Grow early')
    ax.scatter(*gr, s=130, marker='s', color='#52514e', edgecolor=SURF, zorder=5, label='Grid search')
    ax.set_xlabel('Total learner compute (PFLOP)'); ax.set_ylabel('Held-out success (%)')
    ax.set_xlim(0, 0.8); ax.set_ylim(75, 92); ax.legend(loc='lower right', fontsize=11)
    better = [p for p in pts if p[0] <= ge[0] * 1.0001 and p[1] >= ge[1]]
    ax.text(0.01, 0.97, f'Variants: first cut at 0.5–2M steps, keep 1–3, second cut anywhere.\n'
                         f'Variants with ≤ grow-early compute and ≥ its success: {len(better)}',
            transform=ax.transAxes, va='top', fontsize=11, color=INK2)
    save(fig, 'm5_sh_variants.png')


# ---------- M6 smaller grid searches ----------
def m6():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for size in (2, 3, 4):
        for sub in itertools.combinations(FIXED, size):
            c = np.mean([sum(CF[s][k][-1] for k in sub) for s in seeds]) / 1e15
            v = np.mean([tst(BY[s], max(sub, key=lambda k: dev(BY[s], k, T)), T) for s in seeds]) * 100
            ax.scatter(c, v, s=60, color='#52514e', alpha=0.45 + 0.15 * (size - 2), edgecolor='none')
            lab_i = getattr(m6, '_i', 0); m6._i = lab_i + 1
            ax.text(c, v + (1.0 if lab_i % 2 == 0 else -1.0), '{' + ','.join(k[1:] for k in sub) + '}', fontsize=10,
                    color=INK2, ha='center', va='bottom' if lab_i % 2 == 0 else 'top')
    for k in FIXED:
        c = np.mean([CF[s][k][-1] for s in seeds]) / 1e15; v = np.mean([tst(BY[s], k, T) for s in seeds]) * 100
        ax.scatter(c, v, s=70, color=DEPTH_COL[int(k[1:])], edgecolor=SURF, zorder=3)
        ax.text(c + 0.008, v, f'depth {k[1:]}', fontsize=10, color=DEPTH_COL[int(k[1:])], va='center')
    ge = np.mean([ST[s]['Grow early'][0] for s in seeds]) / 1e15, np.mean([ST[s]['Grow early'][1] for s in seeds]) * 100
    ax.scatter(*ge, s=170, color=GROWE_COL, edgecolor=SURF, zorder=5)
    ax.text(ge[0] + 0.01, ge[1] + 0.8, 'Grow early', color=GROWE_COL, fontsize=12, weight='bold')
    ax.set_xlabel('Total learner compute (PFLOP)'); ax.set_ylabel('Held-out success of selected depth (%)')
    ax.set_title('Grid search over every subset of depths {1,2,3,4} (selection by validation)', fontsize=13, color=INK2)
    ax.set_xlim(0, 0.8); ax.set_ylim(60, 92)
    save(fig, 'm6_grid_subsets.png')


# ---------- anytime helpers ----------
def anytime(s):
    by, F = BY[s], CF[s]
    r1, r2 = T // 6, T // 2
    grid = [(sum(F[k][i] for k in FIXED), tst(by, max(FIXED, key=lambda k: dev(by, k, t)), t)) for i, t in enumerate(STEPS)]
    alive2 = sorted(FIXED, key=lambda k: -dev(by, k, r1))[:2]
    win = max(alive2, key=lambda k: dev(by, k, r2))
    sh = []
    for i, t in enumerate(STEPS):
        if t <= r1:
            cost, pool = sum(F[k][i] for k in FIXED), FIXED
        elif t <= r2:
            cost = sum(F[k][IDX[r1]] for k in FIXED) + sum(F[k][i] - F[k][IDX[r1]] for k in alive2); pool = alive2
        else:
            cost = sum(F[k][IDX[r1]] for k in FIXED) + sum(F[k][IDX[r2]] - F[k][IDX[r1]] for k in alive2) + F[win][i] - F[win][IDX[r2]]; pool = [win]
        sh.append((cost, tst(by, max(pool, key=lambda k: dev(by, k, t)), t)))
    one = lambda k: [(F[k][i], tst(by, k, t)) for i, t in enumerate(STEPS)]
    steps_cost = {'Grid search': [4 * t for t in STEPS],
                  'Successive halving': [4 * t if t <= r1 else 4 * r1 + 2 * (t - r1) if t <= r2 else 4 * r1 + 2 * (r2 - r1) + (t - r2) for t in STEPS]}
    return {'Grid search': grid, 'Successive halving': sh, 'Grow early': one('growE'), 'Grow late': one('grow'),
            'Default depth 2': one('d2')}, steps_cost


AT = {s: anytime(s) for s in seeds}


def value_at_budget(curve, b):
    ok = [v for c, v in curve if c <= b * 1.0001]
    return ok[-1] if ok else np.nan  # the model a strategy holds at that spend: its latest evaluation within budget


# ---------- M7 equal budgets ----------
def m7():
    budgets = [0.10, 0.15, 0.18, 0.25, 0.36]
    show = ['Grow early', 'Grow late', 'Successive halving', 'Grid search']
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    w = 0.2
    for j, nm in enumerate(show):
        vals = [np.nanmean([value_at_budget(AT[s][0][nm], b * 1e15) for s in seeds]) * 100 for b in budgets]
        xs = np.arange(len(budgets)) + (j - 1.5) * w
        ax.bar(xs, vals, w * 0.92, color=SC[nm], edgecolor='none', label=nm)
        for x, v in zip(xs, vals):
            if not np.isnan(v): ax.text(x, v + 1, f'{v:.0f}', ha='center', fontsize=10)
    ax.set_xticks(range(len(budgets)), [f'{b:.2f} PFLOP' for b in budgets])
    ax.set_ylabel('Held-out success of the model held (%)'); ax.set_ylim(0, 100); style_axes(ax)
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=11)
    ax.text(0.0, -0.17, 'Value = the model each strategy holds at that spend (latest evaluation within budget). Growth runs end at 0.15–0.18 PFLOP.', transform=ax.transAxes, fontsize=10, color=INK2)
    save(fig, 'm7_equal_budget.png')


# ---------- M8 compute to threshold ----------
def m8():
    thr = [50, 60, 70, 80]
    show = ['Grow early', 'Grow late', 'Default depth 2', 'Successive halving', 'Grid search']
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    w = 0.16
    for j, nm in enumerate(show):
        vals, reach = [], []
        for th in thr:
            cs = []
            for s in seeds:
                hit = [c for c, v in AT[s][0][nm] if v * 100 >= th]
                cs.append(min(hit) / 1e15 if hit else np.nan)
            vals.append(np.nanmean(cs)); reach.append(int(np.sum(~np.isnan(cs))))
        xs = np.arange(len(thr)) + (j - 2) * w
        ax.bar(xs, vals, w * 0.92, color=SC[nm], edgecolor='none', label=nm)
        for x, v, r in zip(xs, vals, reach):
            if r < n: ax.text(x, (0 if np.isnan(v) else v) + 0.01, f'{r}/{n}', ha='center', fontsize=9, color=INK2)
    ax.set_xticks(range(len(thr)), [f'reach {t}%' for t in thr])
    ax.set_ylabel('Learner compute needed (PFLOP)'); style_axes(ax)
    ax.legend(ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
    ax.text(0.0, -0.16, 'Mean over seeds that reach the threshold; k/n marks thresholds not reached by every seed.', transform=ax.transAxes, fontsize=10, color=INK2)
    save(fig, 'm8_compute_to_threshold.png')


# ---------- M9 per-task differences ----------
def m9():
    tasks = BY[seeds[0]]['d1']['state']['task_names']
    def pt(s, k):
        return np.array(BY[s][k]['ev'][-1]['per_task_test']) * 100
    rows = {'vs best depth': [pt(s, 'growE') - pt(s, ST[s]['Best depth'][2]) for s in seeds],
            'vs successive halving': [pt(s, 'growE') - pt(s, ST[s]['Successive halving'][2]) for s in seeds],
            'vs default depth 2': [pt(s, 'growE') - pt(s, 'd2') for s in seeds],
            'vs grow late': [pt(s, 'growE') - pt(s, 'grow') for s in seeds]}
    M = np.array([np.mean(v, 0) for v in rows.values()])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', ['#d95926', '#f3c3ad', '#f2f2f0', '#b7d3f6', '#256abf'])
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.imshow(M, cmap=cmap, vmin=-40, vmax=40, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:+.0f}', ha='center', va='center', fontsize=11, color=INK)
    ax.set_yticks(range(len(rows)), [f'Grow early {k}' for k in rows])
    ax.set_xticks(range(len(tasks)), [TASKS_SHORT(t) for t in tasks], rotation=35, ha='right')
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title('Per-task held-out success difference (points; blue = grow early better)', fontsize=13, color=INK2)
    save(fig, 'm9_per_task_diff.png')


# ---------- M10 validation vs test ----------
def m10():
    xs, ys, cs = [], [], []
    for r in runs:
        for e in r['ev']:
            if e['training_steps'] > 0:
                xs.append(e['success_dev'] * 100); ys.append(e['success_test'] * 100); cs.append(color(r['key']))
    xs, ys = np.array(xs), np.array(ys)
    rr = np.corrcoef(xs, ys)[0, 1]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(xs, ys, s=18, c=cs, alpha=0.6, edgecolor='none')
    ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls='--')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
    ax.set_xlabel('Validation-goal success (%)'); ax.set_ylabel('Held-out test success (%)')
    ax.text(3, 95, f'r = {rr:.2f}, {len(xs)} evaluations\n(all width-400 runs, every 0.5M steps)', fontsize=11, va='top', color=INK2)
    picks_ok = sum(ST[s]['Grid search'][2] == ST[s]['Best depth'][2] for s in seeds)
    ax.text(97, 3, f'grid search picked the\nbest test depth in {picks_ok}/{n} seeds', fontsize=11, ha='right', va='bottom', color=INK2)
    save(fig, 'm10_validation_vs_test.png')


# ---------- M11 anytime in environment steps ----------
def m11():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for nm in ['Grid search', 'Successive halving', 'Grow late', 'Grow early']:
        ys = np.mean([[v for _, v in AT[s][0][nm]] for s in seeds], 0) * 100
        xs = np.array(AT[seeds[0]][1].get(nm, STEPS)) / 1e6
        ax.plot(xs, ys, color=SC[nm], lw=3 if nm == 'Grow early' else 2, marker='o', ms=5, mec=SURF)
        dy = {'Grow early': 4, 'Grow late': -9}.get(nm, 0)
        ax.text(xs[-1] + 0.15, ys[-1] + dy, nm, color=SC[nm], fontsize=12, weight='bold', va='center')
    ax.set_xlabel('Total environment steps, including search (M)'); ax.set_ylabel('Held-out success of selected model (%)')
    ax.set_ylim(0, 100); ax.set_xlim(0, 15)
    save(fig, 'm11_anytime_steps.png')


# ---------- M12 what search picks ----------
def m12():
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    rows = ['Best depth', 'Grid search', 'Successive halving']
    for i, nm in enumerate(rows):
        for j, s in enumerate(seeds):
            k = ST[s][nm][2]; d = int(k[1:])
            ax.add_patch(plt.Rectangle((j - 0.42, len(rows) - 1 - i - 0.38), 0.84, 0.76, color=DEPTH_COL[d]))
            ax.text(j, len(rows) - 1 - i, f'depth {d}\n{ST[s][nm][1] * 100:.0f}%', ha='center', va='center', fontsize=10.5,
                    color='white' if d >= 2 else INK)
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(n), [f'Seed {s}' for s in seeds]); ax.set_yticks(range(len(rows)), rows[::-1])
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    save(fig, 'm12_search_picks.png')


# ---------- E1 train vs held-out, every run ----------
def e1():
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for r in runs_all:
        st = r['state']
        ax.scatter(st['final_success_train'] * 100, st['final_success'] * 100, s=55, color=color(r['key']),
                   marker='o' if r['width'] == 400 else '^', edgecolor=SURF, linewidth=0.8, alpha=0.9)
    ax.plot([40, 100], [40, 100], color=INK2, lw=1, ls='--')
    ax.set_xlim(40, 100); ax.set_ylim(40, 100); ax.set_aspect('equal')
    ax.set_xlabel('Training-goal success (%)'); ax.set_ylabel('Held-out success (%)')
    ax.text(42, 97, f'{len(runs_all)} runs; circles width 400, triangles width 128', fontsize=11, va='top', color=INK2)
    save(fig, 'e1_train_vs_heldout.png')


# ---------- E2 seed variability ----------
def e2():
    ks = FIXED + ['grow', 'growE']
    lab = {'grow': 'Grow late', 'growE': 'Grow early'}
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, k in enumerate(ks):
        v = np.array([BY[s][k]['state']['final_success'] * 100 for s in seeds])
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=color(k), lw=3)
        ax.scatter(i + np.linspace(-0.12, 0.12, n), v, s=55, color=color(k), edgecolor=SURF, zorder=3)
        ax.text(i + 0.3, v.mean(), f'sd {v.std(ddof=1):.1f}', fontsize=10.5, va='center', color=INK2)
    ax.set_xticks(range(len(ks)), [lab.get(k, f'Depth {k[1:]}') for k in ks]); ax.set_ylabel('Final held-out success (%)')
    ax.set_ylim(45, 95); style_axes(ax)
    save(fig, 'e2_seed_variability.png')


# ---------- E3 learning speed ----------
def e3():
    ks = FIXED + ['grow', 'growE']
    lab = {'grow': 'Grow late', 'growE': 'Grow early'}
    thr = [50, 70]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    w = 0.35
    for j, th in enumerate(thr):
        for i, k in enumerate(ks):
            hits = []
            for s in seeds:
                h = [e['training_steps'] for e in BY[s][k]['ev'] if e['success_test'] * 100 >= th]
                hits.append(min(h) / 1e6 if h else np.nan)
            v = np.nanmean(hits); reached = int(np.sum(~np.isnan(hits)))
            x = i + (j - 0.5) * w
            ax.bar(x, v, w * 0.9, color=color(k), alpha=0.55 if j == 0 else 1.0, edgecolor='none')
            ax.text(x, (v if not np.isnan(v) else 0) + 0.05, f'{v:.1f}' + ('' if reached == n else f'\n({reached}/{n})'),
                    ha='center', va='bottom', fontsize=9.5)
    ax.set_xticks(range(len(ks)), [lab.get(k, f'Depth {k[1:]}') for k in ks])
    ax.set_ylabel('Env. steps to first reach threshold (M)'); style_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, alpha=0.55, label='50% held-out'), Patch(color=INK2, label='70% held-out')],
              loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=11)
    ax.set_ylim(0, 3.3)
    save(fig, 'e3_learning_speed.png')


# ---------- E4 growth events ----------
def e4():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.3))
    for k, lab in (('growE', 'Grow early'), ('grow', 'Grow late')):
        r = BY[seeds[0]][k]
        xs, ys = [0], [r['me'][0]['actor_parameters'] / 1e3 if r['me'] else 0]
        for m in r['me']:
            xs.append(m['training_steps'] / 1e6); ys.append(m['actor_parameters'] / 1e3)
        a.step(xs, ys, where='pre', color=color(k), lw=2.5, label=lab)
    for k in FIXED:
        p = BY[seeds[0]][k]['state']['actor_parameters'] / 1e3
        a.axhline(p, color=DEPTH_COL[int(k[1:])], lw=1, ls=':')
        a.text(3.02, p, f'depth {k[1:]}', fontsize=10, va='center', color=DEPTH_COL[int(k[1:])])
    a.set_xlabel('Environment steps (M)'); a.set_ylabel('Actor parameters (thousands)'); a.set_xlim(0, 3.4)
    a.legend(loc='center left', fontsize=11)
    # success change between the evaluation before and after each insertion
    dl = []
    for k in ('growE', 'grow'):
        for s in seeds:
            r = BY[s][k]
            for g in r['state']['grow_at']:
                before, after = at(r, g, 'success_test'), at(r, g + 500_000, 'success_test')
                dl.append((k, g, (after - before) * 100))
    for i, (k, g, d) in enumerate(sorted(dl, key=lambda x: (x[0] != 'growE', x[1]))):
        b.bar(i, d, 0.7, color=color(k), edgecolor='none')
    b.axhline(0, color=INK2, lw=1)
    b.set_xticks([]); b.set_ylabel('Held-out success change over the\n0.5M steps after insertion (points)')
    b.set_xlabel(f'{len(dl)} insertions (grow early left, grow late right)')
    save(fig, 'e4_growth_events.png')


# ---------- E5 hard-task curves ----------
def e5():
    tasks = BY[seeds[0]]['d1']['state']['task_names']
    hard = [t for t in tasks if TASKS_SHORT(t) in ('peg-insert-side', 'push', 'drawer-open', 'pick-place')]
    fig, axes = plt.subplots(1, len(hard), figsize=(14, 3.8), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, t in zip(axes, hard):
        ti = tasks.index(t)
        for k in FIXED + ['grow', 'growE']:
            ys = np.mean([[e['per_task_test'][ti] for e in BY[s][k]['ev']] for s in seeds], 0) * 100
            ax.plot([x / 1e6 for x in STEPS], ys, color=color(k), lw=2.6 if k == 'growE' else 1.6)
        ax.set_title(TASKS_SHORT(t), fontsize=13); ax.set_xlabel('Env. steps (M)'); ax.set_ylim(0, 100)
    axes[0].set_ylabel('Held-out success (%)')
    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[Line2D([], [], color=color(k), lw=2, label={'grow': 'Grow late', 'growE': 'Grow early'}.get(k, f'Depth {k[1:]}'))
                             for k in FIXED + ['grow', 'growE']], fontsize=9.5, loc='upper left')
    save(fig, 'e5_hard_tasks.png')


# ---------- E6 width comparison ----------
def e6():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for w, mk, ls in ((400, 'o', '-'), (128, '^', '--')):
        rs = [r for r in runs_all if r['width'] == w]
        ys = [np.mean([r['state']['final_success'] for r in rs if r['key'] == k]) * 100 for k in FIXED]
        ax.plot(range(1, 5), ys, color=INK2, ls=ls, lw=1.8, marker=mk, ms=8, label=f'Fixed depth, width {w}')
        for k, x in (('growE', 3.15), ('grow', 3.3)):
            v = [r['state']['final_success'] for r in rs if r['key'] == k]
            if v:
                ax.scatter(x, np.mean(v) * 100, s=90, marker=mk, color=color(k), edgecolor=SURF, zorder=3)
    ax.scatter([], [], color=GROWE_COL, label='Grow early (ends at depth 3)'); ax.scatter([], [], color=GROW_COL, label='Grow late (ends at depth 3)')
    ax.set_xticks(range(1, 5), [f'Depth {d}' for d in range(1, 5)]); ax.set_ylabel('Final held-out success (%)')
    ax.set_ylim(50, 95); ax.legend(fontsize=10.5, loc='lower right'); style_axes(ax)
    ax.text(0.01, 0.97, 'width 400: 5 seeds; width 128: 2–3 seeds', transform=ax.transAxes, va='top', fontsize=10, color=INK2)
    save(fig, 'e6_width.png')


for f in (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, e1, e2, e3, e4, e5, e6):
    f()
