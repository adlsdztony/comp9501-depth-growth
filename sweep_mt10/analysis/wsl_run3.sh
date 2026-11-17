#!/usr/bin/env bash
cd ~/growth && mkdir -p runs logs
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '\r ')
.venv/bin/python sweep_mt10/run_one.py --out runs/fixed_w128_d4_s1 --method fixed --depth 4 --width 128 --seed 1 --uuid "$uuid" > logs/e.out 2>&1 &
.venv/bin/python sweep_mt10/run_one.py --out runs/grow_w128_d1_s1 --method grow --depth 1 --width 128 --grow-at 1000000 2000000 --seed 1 --uuid "$uuid" > logs/f.out 2>&1 &
wait
cat logs/e.out logs/f.out
