#!/bin/bash
python main.py --phase train --epoch 40 --dataset_dir flower --checkpoint_dir ./check/flower --sample_dir ./check/flower/sample --gpu 0 --L1_lambda 10
