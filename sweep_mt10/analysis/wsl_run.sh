#!/usr/bin/env bash
cd ~/growth && mkdir -p runs logs
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '\r ')
echo "uuid=$uuid"
.venv/bin/python sweep_mt10/run_one.py --out runs/fixed_w400_d2_s2 --method fixed --depth 2 --seed 2 --uuid "$uuid" > logs/a.out 2>&1 &
.venv/bin/python sweep_mt10/run_one.py --out runs/grow_w400_d1_s2 --method grow --depth 1 --grow-at 1000000 2000000 --seed 2 --uuid "$uuid" > logs/b.out 2>&1 &
wait
cat logs/a.out logs/b.out
