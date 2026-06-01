#!/bin/bash

LOG_DIR="logs_mod"

mkdir -p ${LOG_DIR}

RUN_NAME="lejepa_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

echo "Starting run: ${RUN_NAME}"

CUDA_VISIBLE_DEVICES=0 python3 lejepa.py \
    +lamb=0.4 \
    +V=4 \
    +proj_dim=128 \
    +lr=3e-4 \
    +bs=64 \
    +epochs=1000 \
    +depth=6 \
    +hid_chans=64 \
    2>&1 | tee ${LOG_FILE}

echo "Finished run: ${RUN_NAME}"
