#!/bin/sh

DATASET_DIR="/kaggle/input/generate_image/train"
LABEL_PATH="${DATASET_DIR}/labels.jsonl"
IMAGE_DIR="${DATASET_DIR}/image"
DATASET_LENGTH=10000
TIMEOUT=1000

PROMPT="<image>\nPlease output company, name, email, phone_number, address from business card."

filename="./logs/make_dataset_train.log"

python3 -u make_dataset.py --dataset_dir "${DATASET_DIR}" \
    --label_path "${LABEL_PATH}" \
    --image_dir "${IMAGE_DIR}" \
    --dataset_length "${DATASET_LENGTH}" \
    --timeout "${TIMEOUT}" \
    --prompt "${PROMPT}" > "${filename}" &
