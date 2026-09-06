#!/usr/bin/env bash

export PYTHONPATH=.
python3 main.py train \
    --task pap_ner \
    --run_test \
    --data_dir ./Dataset \
    --model_name_or_path vinai/phobert-base \
    --model_arch crf \
    --output_dir outputs \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --classifier_learning_rate 1e-3 \
    --epochs 20 \
    --early_stop 3 \
    --overwrite_data