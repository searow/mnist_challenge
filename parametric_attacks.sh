#!/usr/bin/env bash

ks=( 16 32 64 128 256 )
top_grads=( 1 2 4 8 16 32 64 )
as=( 0.01 0.02 0.04 0.08 0.16 0.32 )

for k in ${ks[@]}; do
  for grad in ${top_grads[@]}; do
    for a in ${as[@]}; do
      python3 partial_fgsm.py --top_grads=$grad --k=$k --a=$a --delete_attacks=true
    done
  done
done

