#!/bin/bash

cls=($(seq 154 157))
cls+=(208 249 258)
len_cls=${#cls[*]}

echo $len_cls

now_idx=0

while [ $now_idx -lt $len_cls ]; do
    python sample.py --image-size 256 --seed 2 --base_samples 'refs/cat' --down_N 4 --range_t 80 --cls_labels $((${cls[$now_idx]})) $((${cls[$now_idx]})) $((${cls[$now_idx]})) $((${cls[$now_idx]}))
    mv output/N4_t80.png output/N4_t80_$((${cls[$now_idx]})).png
    now_idx=$(($now_idx+1))
done