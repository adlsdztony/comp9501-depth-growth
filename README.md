# When should a reinforcement learning policy grow deeper?

## Depth-sweep study (2026-11-17)

Question: when the right actor depth is unknown, can one run that grows depth
(function-preserving identity insertion) match a searched fixed depth at lower cost?

Meta-World MT10, PPO + PopArt, actor width 400, 3M environment steps per run,
per task 50 goal variants split 30 train / 10 validation / 10 held-out test.
43 completed runs (5 seeds at width 400, 2 at width 128).

Held-out success, width 400, paired over 5 seeds (mean):

| Strategy | Success | Total env steps |
| --- | ---: | ---: |
| Best fixed depth (hindsight) | 89% | 3M |
| Grid search over depth 1–4 | 89% | 12M |
| Successive halving (simulated from logged curves) | 85% | 5.5M |
| Grow early, depth 1→3 at 0.5M/1M | 84% | 3M |
| Default depth 2 | 81% | 3M |
| Random depth | 79% | 3M |
| Grow late, depth 1→3 at 1M/2M | 79% | 3M |

- Depth matters (depth 1: 64%, depth 4: 88%); held-out goals are not harder than training goals.
- Growing early comes close to successive halving at 55% of its cost, but stays 5 points below the best depth.
- At width 128 (n=2) growth did not help.
- Learner FLOPs scale 2.3× from depth 1 to 4, but measured GPU time barely changes (overhead- and load-bound), so costs are compared in environment steps and FLOPs.

Exploratory: fixed growth schedules (not the GKP controller), few seeds, one benchmark.

| Path | Content |
| --- | --- |
| `deck/COMP9501_Growth_Results_v8.pptx` | Latest deck: v7 with a related-work comparison matrix and a contribution slide. v7 = v6 plus related-work and problem-to-approach slides. v6 = v5 plus benchmark (MT10) and algorithm (PPO) slides. v5 = v4 plus 18 backup analysis slides. Earlier versions: v1 (minimal), v2 (detailed), v3 (concise). Each `deck/deck*.json` is the content source of its deck, with speaker notes |
| `report/` | NeurIPS-format report: `main.tex`, `refs.bib`, `neurips_2026.sty`, compiled `main.pdf` |
| `figures/` | All rendered figures; `figures/w400/summary.json`, `figures/w128/summary.json` hold the aggregated numbers |
| `results/all/` | One directory per run: `state.json`, `evaluations.jsonl` (train/dev/test success every 500k steps), `metrics.jsonl`, `growth_events.jsonl`, logs. Checkpoints not included. One per (width, config, seed) is used; a duplicate depth-4 seed-2 run is resolved by `aggregate.py` |
| `sweep_mt10/` | Runner derived from `full_mt10` (`sweep_worker.py`, `run_one.py`, `modal_job.py`) |
| `sweep_mt10/analysis/` | `aggregate.py` (data figures + summaries), `static_figs.py` (schematics, goal split, pilot), `extra_figs.py` (backup analyses into `figures/backup/`), `build_deck.py` (v1 deck), `build_deck_v2.py` (v2–v4 decks), `fetch.sh` (pull results from all hosts), `wsl_run*.sh` (local launch helpers) |

Reproduce figures, decks and report (Python with `matplotlib`, `numpy`, `python-pptx`, `Pillow`):

```bash
python sweep_mt10/analysis/aggregate.py results/all figures
python sweep_mt10/analysis/static_figs.py figures/static
python sweep_mt10/analysis/extra_figs.py results/all figures
python sweep_mt10/analysis/build_deck.py figures deck/COMP9501_Growth_Results.pptx deck/deck.json
python sweep_mt10/analysis/build_deck_v2.py figures deck/COMP9501_Growth_Results_v2.pptx deck/deck_v2.json
python sweep_mt10/analysis/build_deck_v2.py figures deck/COMP9501_Growth_Results_v3.pptx deck/deck_v3.json
python sweep_mt10/analysis/build_deck_v2.py figures deck/COMP9501_Growth_Results_v4.pptx deck/deck_v4.json
python sweep_mt10/analysis/build_deck_v2.py figures deck/COMP9501_Growth_Results_v5.pptx deck/deck_v5.json
python sweep_mt10/analysis/build_deck_v2.py figures deck/COMP9501_Growth_Results_v6.pptx deck/deck_v6.json
python sweep_mt10/analysis/build_deck_v2.py figures deck/COMP9501_Growth_Results_v7.pptx deck/deck_v7.json
python sweep_mt10/analysis/build_deck_v2.py figures deck/COMP9501_Growth_Results_v8.pptx deck/deck_v8.json
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Committed decks were built from the figures at the time of each version; rebuilding an older deck uses the current figures, which have since changed (FLOP label precision, depth 2 removed from `anytime_flops.png`).

Run one configuration (Linux, environment from `full_mt10/resolved.lock`):

```bash
python sweep_mt10/run_one.py --out runs/grow_s1 --method grow --depth 1 --grow-at 500000 1000000 --seed 1 --uuid GPU-xxxx
```

---

## Original exploration kit

Independent exploration kit for COMP9501. Start with **Research-Brief.pdf**,
then run the CPU smoke before starting GPU experiments. This bundle concerns
the **state-based Meta-World policy**, not the separate vision-language pilot.

## What is here

| Item | Status |
| --- | --- |
| `pilot_3task/` + `evidence/` | Completed one-seed, three-task insertion diagnostic; source and lightweight results. The launcher substitutes a local GPU UUID. |
| `full_mt10/` | Frozen full-MT10 timing runner and dependencies, with original SHA-256 source pins. A CPU smoke ran 1,000 Meta-World steps and 64 Adam updates; no full study outcomes are included. |
| `launch.sh` | Local wrapper for an isolated CPU smoke, three-task reproduction, or full timing run. One fresh output directory per invocation. |

No checkpoints are included: either GPU run first trains its own fresh parent.
A smoke run is a real short environment/optimizer test, not a performance
result or a toy learning benchmark.

## Quickstart (Linux x86-64)

Use Python **3.12**, GNU `taskset`, and at least 12 accessible logical CPUs.
Install the hash-pinned dependencies in a fresh environment. The validated
stack uses PyTorch 2.8, Meta-World 3.0, MuJoCo 3.7 and Tianshou 2.0.
The dependency lock targets Linux x86-64. Other platforms are not verified.
The PyTorch lock includes CUDA runtime components even for CPU smoke; allow
disk space and installation time. A suitable NVIDIA driver is required for
GPU runs. Use an idle GPU you are allowed to use, and leave 45 GiB free.
Weights/services/accounts are not needed; tracking is disabled.

```bash
# From the extracted kit. UV is optional; pip is shown here.
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r full_mt10/resolved.lock
(cd full_mt10 && sha256sum -c SOURCE_PINS.sha256)
# Pick 12 allowed logical CPUs; do not assume 0-11 are permitted in a container.
taskset -pc $$
CPUS=0-11  # replace as needed
bash launch.sh smoke .venv/bin/python "$CPUS" smoke-17711
# Inspect smoke-17711/runs/parent/state.json and parent.log.
```

If the pinned wheel is unavailable on your platform, use the separate
`requirements.in` to create a platform-specific environment and record the
new dependency versions. That is exploratory rather than exact reproduction.

### Optional GPU training

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
GPU_UUID=GPU-REPLACE-WITH-ONE-ID
bash launch.sh pilot .venv/bin/python "$CPUS" three-task-17701 "$GPU_UUID"
bash launch.sh full .venv/bin/python "$CPUS" full-mt10-17711 "$GPU_UUID" 17711
# Repeat full for seeds 17712 and 17713, using new output directories.
```

These are separate runs. On a slower machine the wall deadlines may prevent
endpoints; retain the stopped run as incomplete instead of reducing budgets
silently. Pilot caps are 999,990 parent transitions and 299,997 per arm,
with a 3,000 s outer deadline. Full caps are 3,000,000 for the parent and
1,000,000 for each arm, with a 5,300 s outer deadline. Evaluation and setup
also use wall budget. Read each `runs/*/state.json` and aggregate only
verified endpoints. Saved `summary.json` and `summary.csv` are written only
after the checks succeed. Checkpoint loads are for locally generated, trusted
files only.

## 简明中文版

- 先读 `Research-Brief.pdf`，再按以上命令在 Linux / Python 3.12
  建立环境并跑 CPU smoke。请选择实际可用的 12 个逻辑核。
- `pilot_3task` 有三任务单种子结果；`full_mt10` 提供完整十任务的
  对照流程和代码，本包不含该研究的最终数据。
- GPU 只用获得许可且空闲的一张卡。每次指定新目录。结果看
  `summary.csv` 和 `runs/*/state.json`；超时不属于完成结果。
- 研究比较保持原深度、立即扩层和延迟扩层；扩层前后的输出等价
  不意味着后续训练一定更好。

## Provenance and interpretation

Full common archive SHA-256: `ec442938a54a7a1d9ef0e3c1bb6e18fb437e7755b5afab5a5d25678226780f10`.
The pilot code matches the completed run except that its orchestrator accepts
a local GPU UUID. The run wrapper provisions the pinned learner by local
reference to the copy already included in the ZIP. No active run directories
or machine-specific launch logs are included. See `THIRD_PARTY_NOTICES.md`.
