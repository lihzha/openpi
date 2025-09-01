#!/bin/bash
set -euo pipefail

TPU_NAME="pi0"
ZONE="us-central2-b"
TYPE="v4"
TOPOLOGY="2x4x4"
VERSION="tpu-ubuntu2204-base"

while true; do
  echo "Attempting to create TPU: $TPU_NAME ..."
  if gcloud compute tpus tpu-vm create "$TPU_NAME" \
      --zone="$ZONE" \
      --type="$TYPE" \
      --topology="$TOPOLOGY" \
      --version="$VERSION"; then
    echo "✅ TPU $TPU_NAME created successfully!"
    break
  else
    echo "❌ Failed to create TPU $TPU_NAME. Retrying in 0s..."
    # sleep 1
  fi
done
