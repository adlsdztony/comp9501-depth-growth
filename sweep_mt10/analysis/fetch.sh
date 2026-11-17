#!/usr/bin/env bash
# Pull lightweight run outputs (json/jsonl/log, no checkpoints) from all hosts into results/.
set -u
cd "$(dirname "$0")/.."
R=../results; mkdir -p $R/lab $R/pro16 $R/wsl $R/modal
pull_ssh() { # host remote_glob_dirs... -> local subdir
  host=$1; dest=$2; shift 2
  ssh $host "cd / && tar czf - --exclude='*.pt' --exclude='reference' --exclude='__pycache__' --exclude='*.py' --exclude='*.lock' $* 2>/dev/null" | tar xzf - -C $R/$dest 2>/dev/null
}
pull_ssh lab lab home/zlzhou/workspace/growth/runs home2/zlzhou/growth_runs
pull_ssh pro16 pro16 home/adlsdztony/growth/runs
wsl -d Ubuntu -- bash -lc "cd / && tar czf - --exclude='*.pt' --exclude='reference' --exclude='*.py' --exclude='*.lock' home/zzl07/growth/runs 2>/dev/null" | tar xzf - -C $R/wsl 2>/dev/null
eval "$(cloud-accounts env modal personal --shell bash)"; export PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
mkdir -p $R/modal_tgz
for f in $(modal volume ls depth-sweep-results / 2>/dev/null | grep -o "[a-zA-Z_0-9-]*\.tgz"); do
  [ -s "$R/modal_tgz/$f" ] || (cd $R/modal_tgz && modal volume get --force depth-sweep-results $f . >/dev/null 2>&1)
done
for t in $R/modal_tgz/*.tgz; do [ -f "$t" ] && tar xzf "$t" -C $R/modal; done
# flatten: one dir per run under results/all
rm -rf $R/all; mkdir -p $R/all
find $R/lab $R/pro16 $R/wsl $R/modal -name state.json -path '*/runs/*/state.json' | while read s; do
  rundir=$(dirname "$(dirname "$(dirname "$s")")"); name=$(basename "$(dirname "$s")")
  host=$(echo "$rundir" | cut -d/ -f3)
  [ -e "$R/all/${host}_$name" ] || ln -s "$(cd "$rundir" && pwd)" "$R/all/${host}_$name" 2>/dev/null || cp -r "$rundir" "$R/all/${host}_$name"
done
ls $R/all | wc -l
