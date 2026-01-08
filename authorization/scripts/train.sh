#!/bin/bash
set -e
trap "echo '[!] 任务被中断'; exit 1" INT

CUDA_VISIBLE_DEVICES=1 python main.py \
--device cuda:0 \
--gamma 0.5 

