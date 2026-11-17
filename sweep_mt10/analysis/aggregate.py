"""Aggregate depth-sweep / growth runs into figures and a summary table.

usage: python aggregate.py RUNS_DIR OUT_DIR
RUNS_DIR contains one directory per run (each with runs/<run_id>/{state.json,evaluations.jsonl,metrics.jsonl}).
Only runs with status == completed enter final-endpoint statistics; partial runs are plotted up to their last eval.
"""
import json, sys, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
DEPTH_COL = {1: '#86b6ef', 2: '#5598e7', 3: '#256abf', 4: '#0d366b'}  # ordinal blue ramp
GROW_COL = '#eb6834'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False,
    'lines.linewidth': 2.2, 'axes.titleweight': 'bold', 'axes.titlesize': 15,
})

CRITIC_PARAMS = None


def load():
    runs = []
    for d in sorted(RUNS.iterdir()):
        for r in (d / 'runs').glob('*') if (d / 'runs').exists() else []:
            sp = r / 'state.json'
            if not sp.exists() or not (r / 'evaluations.jsonl').exists():
                continue
            s = json.loads(sp.read_text())
            if s.get('phase') != 'development':
                continue
            ev = [json.loads(l) for l in (r / 'evaluations.jsonl').read_text().splitlines() if l.strip()]
            me = [json.loads(l) for l in (r / 'metrics.jsonl').read_text().splitlines() if l.strip()] if (r / 'metrics.jsonl').exists() else []
            key = ('growE' if min(s['grow_at']) < 1_000_000 else 'grow') if s['method'] == 'grow' else f"d{s['start_depth']}"
            runs.append(dict(dir=d.name, key=key, width=s['width'], seed=s['seed'], state=s, ev=ev, me=me,
                             done=s['status'] == 'completed'))
    # one run per (width, config, seed): duplicates (a re-run of a Modal job that later also finished) keep the first by host order
    seen, out = set(), []
    for r in sorted(runs, key=lambda r: (not r['done'], r['dir'])):
        k = (r['width'], r['key'], r['seed'])
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def flops(run):
    """Analytic learner FLOPs: rollout forward (2P/sample) + 16 PPO epochs (6P/sample), actor+critic."""
    crit = run['state'].get('critic_parameters', 0)
    total = 0.
    for m in run['me']:
        p = m.get('actor_parameters', run['state']['actor_parameters']) + crit
        total += 100_000 * (2 * p + 16 * 6 * p)
    return total


GROWE_COL = '#6250d6'
GLAB = {'grow': 'Growth 1→3 @1M,2M', 'growE': 'Growth 1→3 @0.5M,1M'}


def label(key):
    return GLAB[key] if key in GLAB else f'Fixed depth {key[1:]}'


def color(key):
    return GROW_COL if key == 'grow' else GROWE_COL if key == 'growE' else DEPTH_COL[int(key[1:])]


def isgrow(k):
    return k.startswith('grow')


def keys_of(runs):
    ks = sorted({r['key'] for r in runs if not isgrow(r['key'])}, key=lambda k: int(k[1:]))
    return ks + [g for g in ('grow', 'growE') if any(r['key'] == g for r in runs)]


def curves(runs, metric='success_test'):
    runs = [r for r in runs if r['done']]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for k in keys_of(runs):
        rs = [r for r in runs if r['key'] == k]
        steps = sorted({e['training_steps'] for r in rs for e in r['ev']})
        mean, lo, hi, xs = [], [], [], []
        for st in steps:
            v = [e[metric] for r in rs for e in r['ev'] if e['training_steps'] == st and metric in e]
            if len(v) < max(1, len(rs) - 0):
                continue
            xs.append(st / 1e6); mean.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
        if not xs:
            continue
        ax.plot(xs, np.array(mean) * 100, color=color(k), label=f'{label(k)} (n={len(rs)})',
                lw=2.6 if isgrow(k) else 2.0)
        ax.fill_between(xs, np.array(lo) * 100, np.array(hi) * 100, color=color(k), alpha=0.07, lw=0)

    ax.set_xlabel('Environment steps (M)')
    ax.set_ylabel({'success_test': 'Success on held-out goals (%)', 'success_train': 'Success on training goals (%)',
                   'success_dev': 'Success on validation goals (%)'}[metric])
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f'curves_{metric}.png', dpi=200); plt.close(fig)


def final_by_depth(runs):
    fin = [r for r in runs if r['done']]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ks = keys_of(fin)
    rows = []
    for i, k in enumerate(ks):
        rs = [r for r in fin if r['key'] == k]
        te = [r['state']['final_success'] * 100 for r in rs]
        tr = [r['state']['final_success_train'] * 100 for r in rs]
        rows.append((k, np.mean(tr), np.mean(te), len(rs)))
        ax.bar(i - 0.19, np.mean(tr), 0.36, color=color(k), alpha=0.35, edgecolor='none')
        ax.bar(i + 0.19, np.mean(te), 0.36, color=color(k), edgecolor='none')
        ax.scatter([i + 0.19] * len(te), te, s=28, color=INK, zorder=3)
        ax.text(i + 0.19, max(te) + 2.5, f'{np.mean(te):.0f}', ha='center', color=INK, fontsize=12)
    ax.set_xticks(range(len(ks)), [{'grow': 'Grow\n@1M,2M', 'growE': 'Grow\n@0.5M,1M'}.get(k, f'Depth {k[1:]}') for k in ks])
    ax.set_ylabel('Final success (%)'); ax.set_ylim(0, 100)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, alpha=0.35, label='Training goals'), Patch(color=INK2, label='Held-out goals')],
              loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=11, ncol=2)
    ax.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(OUT / 'final_by_depth.png', dpi=200); plt.close(fig)
    return rows


def at(run, step, metric):
    v = [e[metric] for e in run['ev'] if e['training_steps'] == step]
    return v[0] if v else None


REF_GPU = {}


def set_ref_gpu(runs):
    """GPU time differs by card; use each config's mean learner GPU time on the A6000 runs."""
    REF_GPU.clear()
    for k in keys_of(runs):
        v = [r['state']['learner_gpu_s'] for r in runs if r['key'] == k and r['done'] and 'A6000' in r['state'].get('cuda_name', '')]
        if v:
            REF_GPU[k] = float(np.mean(v))


def cost_frontier(runs):
    set_ref_gpu(runs)
    """Per seed: cost (env steps, learner GPU-s, FLOPs) and test success of each strategy."""
    fin = [r for r in runs if r['done']]
    fixed_keys = sorted({r['key'] for r in fin if not isgrow(r['key'])}, key=lambda k: int(k[1:]))
    seeds = sorted({r['seed'] for r in fin})
    T = max(r['state']['training_steps'] for r in fin)
    res = defaultdict(list)
    for s in seeds:
        by = {r['key']: r for r in fin if r['seed'] == s}
        if not all(k in by for k in fixed_keys):
            continue
        K = len(fixed_keys)
        gpu = {k: REF_GPU.get(k, by[k]['state']['learner_gpu_s']) for k in by}
        fl = {k: flops(by[k]) for k in by}
        # Grid search: run all K to the end, select by final dev success, report its test success.
        best = max(fixed_keys, key=lambda k: (by[k]['state']['final_success_dev'], -int(k[1:])))
        res['Grid search (4 runs)'].append(dict(steps=K * T, gpu=sum(gpu[k] for k in fixed_keys), flops=sum(fl[k] for k in fixed_keys),
                                       test=by[best]['state']['final_success'], pick=best))
        # Successive halving (simulated from logged curves): rungs at T/6 (keep half), T/2 (keep 1).
        r1, r2 = T // 6, T // 2
        alive = sorted(fixed_keys, key=lambda k: -(at(by[k], r1, 'success_dev') or 0))[:max(1, K // 2)]
        win = max(alive, key=lambda k: at(by[k], r2, 'success_dev') or 0)
        frac = lambda k, a, b: (b - a) / T
        st = K * r1 + len(alive) * (r2 - r1) + (T - r2)
        g = sum(gpu[k] * r1 / T for k in fixed_keys) + sum(gpu[k] * (r2 - r1) / T for k in alive) + gpu[win] * (T - r2) / T
        f = sum(fl[k] * r1 / T for k in fixed_keys) + sum(fl[k] * (r2 - r1) / T for k in alive) + fl[win] * (T - r2) / T
        res['Successive halving'].append(dict(steps=st, gpu=g, flops=f, test=by[win]['state']['final_success'], pick=win))
        # Oracle: best fixed depth in hindsight (test), cost of one run (unattainable without search).
        orc = max(fixed_keys, key=lambda k: by[k]['state']['final_success'])
        res['Best depth (oracle)'].append(dict(steps=T, gpu=gpu[orc], flops=fl[orc], test=by[orc]['state']['final_success'], pick=orc))
        res['Random depth'].append(dict(steps=T, gpu=np.mean([gpu[k] for k in fixed_keys]), flops=np.mean([fl[k] for k in fixed_keys]),
                                        test=np.mean([by[k]['state']['final_success'] for k in fixed_keys]), pick='avg'))
        # Default guess: the original recipe's depth 2.
        if 'd2' in by:
            res['Default depth 2'].append(dict(steps=T, gpu=gpu['d2'], flops=fl['d2'], test=by['d2']['state']['final_success'], pick='d2'))
        for g, nm in (('grow', 'Growth @1M,2M'), ('growE', 'Growth @0.5M,1M')):
            if g in by:
                res[nm].append(dict(steps=T, gpu=gpu[g], flops=fl[g], test=by[g]['state']['final_success'], pick=g))
    style = {'Grid search (4 runs)': ('#52514e', 's'), 'Successive halving': ('#008300', 'D'), 'Best depth (oracle)': ('#0d366b', '*'),
             'Random depth': ('#86b6ef', 'o'), 'Growth @1M,2M': (GROW_COL, 'o'), 'Growth @0.5M,1M': (GROWE_COL, 'o')}
    for axis, xl, fn in [('steps', 'Total environment steps, including search (M)', lambda v: v / 1e6),
                         ('gpu', 'Total learner GPU time, including search (A6000 min)', lambda v: v / 60),
                         ('flops', 'Total learner compute, including search (PFLOP)', lambda v: v / 1e15)]:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        pts = []
        for name, rows in res.items():
            if not rows or name not in style:
                continue
            c, m = style[name]
            xs = np.array([fn(r[axis]) for r in rows]); ys = np.array([r['test'] * 100 for r in rows])
            se = ys.std(ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else 0
            ax.errorbar(xs.mean(), ys.mean(), yerr=se, color=c, lw=1.6, capsize=0, zorder=2)
            ax.scatter([xs.mean()], [ys.mean()], color=c, marker=m, s=170 if m != '*' else 360,
                       edgecolor=SURF, linewidth=1.5, zorder=3)
            pts.append([xs.mean(), ys.mean(), name, c])
        # direct labels, de-collided vertically within each x column
        xmax = max(p[0] for p in pts)
        for col in sorted({round(p[0], 1) for p in pts}):
            grp = sorted([p for p in pts if round(p[0], 1) == col], key=lambda p: p[1])
            ly = []
            for p in grp:
                y = p[1] if not ly else max(p[1], ly[-1] + 3.2)
                ly.append(y)
            for p, y in zip(grp, ly):
                right = p[0] < 0.8 * xmax
                ax.text(p[0] + (0.025 if right else -0.025) * xmax, y, p[2], va='center', ha='left' if right else 'right',
                        fontsize=12, color=INK)
        ax.set_xlabel(xl); ax.set_ylabel('Held-out success (%)')
        ax.set_ylim(55, 100); ax.set_xlim(0, xmax * 1.08)
        ax.grid(axis='x', visible=False)
        fig.tight_layout(); fig.savefig(OUT / f'cost_{axis}.png', dpi=200); plt.close(fig)
    return res


def strategy_panel(res, axis='steps', unit=1e6, xl='Total environment steps (M)', fname='strategy_panel.png'):
    order = ['Best depth (oracle)', 'Grid search (4 runs)', 'Successive halving', 'Growth @0.5M,1M', 'Growth @1M,2M',
             'Default depth 2', 'Random depth']
    cols = {'Best depth (oracle)': '#0d366b', 'Grid search (4 runs)': '#52514e', 'Successive halving': '#008300',
            'Growth @0.5M,1M': GROWE_COL, 'Growth @1M,2M': GROW_COL, 'Default depth 2': '#5598e7', 'Random depth': '#86b6ef'}
    names = {'Best depth (oracle)': 'Best fixed depth (hindsight)', 'Grid search (4 runs)': 'Grid search, 4 depths',
             'Successive halving': 'Successive halving', 'Growth @0.5M,1M': 'Grow early (0.5M, 1M)',
             'Growth @1M,2M': 'Grow late (1M, 2M)', 'Default depth 2': 'Default depth 2', 'Random depth': 'Random depth'}
    rows = [k for k in order if res.get(k)]
    n = len(res[rows[0]])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 0.62 * len(rows) + 1.4), sharey=True,
                               gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.08))
    for i, k in enumerate(rows):
        y = len(rows) - 1 - i
        t = np.array([r['test'] for r in res[k]]) * 100
        se = t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else 0
        a.errorbar(t.mean(), y, xerr=se, fmt='o', color=cols[k], ms=11, lw=2, capsize=0, mec=SURF, mew=1.5)
        a.text(t.mean() + se + 0.5, y, f'{t.mean():.0f}', color=INK, fontsize=12, va='center')
        c = np.mean([r[axis] for r in res[k]]) / unit
        b.barh(y, c, 0.5, color=cols[k], edgecolor='none')
        cmax = max(np.mean([r[axis] for r in res[kk]]) / unit for kk in rows)
        b.text(c + 0.012 * cmax, y, f'{c:.2f}' if cmax < 2 else (f'{c:.1f}' if c < 10 else f'{c:.0f}'),
               va='center', fontsize=12, color=INK)
    a.set_yticks(range(len(rows)), [names[k] for k in rows][::-1])
    a.set_xlabel(f'Held-out success (%)   mean ± s.e., n={n} seeds'); lo = min(np.mean([r['test'] for r in res[k]]) for k in rows) * 100
    a.set_xlim(5 * np.floor((lo - 6) / 5), 95)
    b.set_xlabel(xl); b.set_xlim(0, max(np.mean([r[axis] for r in res[k]]) / unit for k in rows) * 1.18)
    for ax in (a, b):
        ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    b.spines['left'].set_visible(False)
    a.axhline(len(rows) - 1.5, color=GRID, lw=1)
    fig.savefig(OUT / fname, dpi=200, bbox_inches='tight', pad_inches=0.15); plt.close(fig)


def compute_vs_gpu(runs):
    fin = [r for r in runs if r['done'] and 'A6000' in r['state'].get('cuda_name', '')]
    ks = keys_of(fin)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(wspace=0.3))
    lab = [{'grow': 'Grow\nlate', 'growE': 'Grow\nearly'}.get(k, f'D{k[1:]}') for k in ks]
    for i, k in enumerate(ks):
        rs = [r for r in fin if r['key'] == k]
        f = np.mean([flops(r) for r in rs]) / 1e15
        g = [r['state']['learner_gpu_s'] / 60 for r in rs]
        a.bar(i, f, 0.6, color=color(k), edgecolor='none')
        b.bar(i, np.mean(g), 0.6, color=color(k), edgecolor='none')
        b.scatter([i] * len(g), g, s=18, color=INK, zorder=3)
    a.set_title('Learner compute (PFLOP)', fontsize=14); b.set_title('Measured GPU time (A6000 min)', fontsize=14)
    for ax in (a, b):
        ax.set_xticks(range(len(ks)), lab); ax.grid(axis='x', visible=False)
    fig.savefig(OUT / 'compute_vs_gpu.png', dpi=200, bbox_inches='tight', pad_inches=0.15); plt.close(fig)


def flops_per_run(runs):
    fin = [r for r in runs if r['done']]
    ks = keys_of(fin)
    lab = [{'grow': 'Grow late\n1→3', 'growE': 'Grow early\n1→3'}.get(k, f'Depth {k[1:]}') for k in ks]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, k in enumerate(ks):
        f = np.mean([flops(r) for r in fin if r['key'] == k]) / 1e15
        ax.bar(i, f, 0.6, color=color(k), edgecolor='none')
        ax.text(i, f + 0.004, f'{f:.2f}', ha='center', va='bottom', fontsize=12, color=INK)
    ax.set_xticks(range(len(ks)), lab); ax.grid(axis='x', visible=False)
    ax.set_ylabel('Learner compute per 3M-step run (PFLOP)')
    fig.savefig(OUT / 'flops_per_run.png', dpi=200, bbox_inches='tight', pad_inches=0.15); plt.close(fig)


def per_task(runs):
    fin = [r for r in runs if r['done']]
    ks = keys_of(fin)
    tasks = fin[0]['state']['task_names']
    M = []
    for k in ks:
        rows = [r['ev'][-1]['per_task_test'] for r in fin if r['key'] == k and r['ev'][-1]['stream'] == 'final']
        M.append(np.mean(rows, 0) * 100)
    M = np.array(M)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fd', '#b7d3f6', '#5598e7', '#1c5cab', '#0d366b'])
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(ks) + 1.8))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=11, color='white' if M[i, j] > 60 else INK)
    ax.set_yticks(range(len(ks)), [{'grow': 'Grow late', 'growE': 'Grow early'}.get(k, f'Depth {k[1:]}') for k in ks])
    ax.set_xticks(range(len(tasks)), [t.replace('-v3', '').replace('-v2', '') for t in tasks], rotation=35, ha='right')
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.savefig(OUT / 'per_task.png', dpi=200, bbox_inches='tight', pad_inches=0.15); plt.close(fig)


def growth_traj(runs):
    fin = [r for r in runs if r['done']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, g, ref, title in [(axes[0], 'growE', 'd3', 'Grow early: +layer at 0.5M, 1M'), (axes[1], 'grow', 'd3', 'Grow late: +layer at 1M, 2M')]:
        for r in [r for r in fin if r['key'] == 'd1']:
            ax.plot([e['training_steps'] / 1e6 for e in r['ev']], [e['success_test'] * 100 for e in r['ev']], color=DEPTH_COL[1], lw=1.2, alpha=0.6)
        for r in [r for r in fin if r['key'] == ref]:
            ax.plot([e['training_steps'] / 1e6 for e in r['ev']], [e['success_test'] * 100 for e in r['ev']], color=DEPTH_COL[3], lw=1.2, alpha=0.6)
        for r in [r for r in fin if r['key'] == g]:
            ax.plot([e['training_steps'] / 1e6 for e in r['ev']], [e['success_test'] * 100 for e in r['ev']], color=color(g), lw=2.2)
        for x in (r['state']['grow_at'] for r in fin if r['key'] == g):
            for xx in x:
                ax.axvline(xx / 1e6, color=color(g), lw=1, ls=':')
            break
        ax.set_title(title, fontsize=14, color=INK); ax.set_xlabel('Environment steps (M)'); ax.set_ylim(0, 100)
    axes[0].set_ylabel('Held-out success (%)')
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color=DEPTH_COL[1], lw=1.5, label='Fixed depth 1 (per seed)'),
                            Line2D([], [], color=DEPTH_COL[3], lw=1.5, label='Fixed depth 3 (per seed)'),
                            Line2D([], [], color=INK2, lw=2.2, label='Growth run (per seed)')], loc='lower right', fontsize=11)
    fig.savefig(OUT / 'growth_traj.png', dpi=200, bbox_inches='tight', pad_inches=0.15); plt.close(fig)


def main():
    runs = load()
    print(f'{len(runs)} runs, {sum(r["done"] for r in runs)} completed')
    for w in sorted({r['width'] for r in runs}):
        sub = [r for r in runs if r['width'] == w]
        global OUT
        base = OUT
        OUT = base / f'w{w}'; OUT.mkdir(exist_ok=True)
        for m in ('success_test', 'success_train', 'success_dev'):
            curves(sub, m)
        rows = final_by_depth(sub) if any(r['done'] for r in sub) else []
        res = cost_frontier(sub) if any(r['done'] for r in sub) else {}
        compute_vs_gpu(sub)
        per_task(sub); growth_traj(sub); flops_per_run(sub)
        if res:
            strategy_panel(res)
            strategy_panel(res, 'flops', 1e15, 'Total learner compute (PFLOP)', 'strategy_panel_flops.png')
        summary = dict(width=w, runs=[dict(dir=r['dir'], key=r['key'], seed=r['seed'], status=r['state']['status'],
                                            steps=r['state']['training_steps'], final_test=r['state'].get('final_success'),
                                            final_train=r['state'].get('final_success_train'), final_dev=r['state'].get('final_success_dev'),
                                            learner_gpu_s=r['state'].get('learner_gpu_s'), elapsed_s=r['state'].get('elapsed_s'),
                                            flops=flops(r)) for r in sub],
                       final_by_key=[dict(key=k, train=a, test=b, n=n) for k, a, b, n in rows],
                       strategies={k: dict(n=len(v), test=float(np.mean([x['test'] for x in v])), steps=float(np.mean([x['steps'] for x in v])),
                                           gpu_s=float(np.mean([x['gpu'] for x in v])), flops=float(np.mean([x['flops'] for x in v])),
                                           picks=[x['pick'] for x in v]) for k, v in res.items() if v})
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary['final_by_key'], indent=1), json.dumps(summary['strategies'], indent=1))
        OUT = base


if __name__ == '__main__':
    main()
