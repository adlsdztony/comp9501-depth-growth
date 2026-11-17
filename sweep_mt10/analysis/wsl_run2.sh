#!/usr/bin/env bash
cd ~/growth && mkdir -p runs logs
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '\r ')
.venv/bin/python sweep_mt10/run_one.py --out runs/grow_w400_d1_s4 --method grow --depth 1 --grow-at 1000000 2000000 --seed 4 --uuid "$uuid" > logs/c.out 2>&1 &
.venv/bin/python sweep_mt10/run_one.py --out runs/growE_w400_d1_s4 --method grow --depth 1 --grow-at 500000 1000000 --seed 4 --uuid "$uuid" > logs/d.out 2>&1 &
wait
cat logs/c.out logs/d.out
