# Override fsdp_devices, batch_size, data.shuffle_buffer_size, data.summation_steps
uv run --group rlds scripts/train.py pi0_droid_cot_v6 \
  --exp-name=my_run --overwrite \
  --fsdp-devices=8 \
  --batch-size=128 \
  --data.shuffle-buffer-size=300000 \
  --data.summation-steps=12


# Another example with v4
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group rlds scripts/train.py pi0_droid_cot_v4 \
  --exp-name=v4_bs256_lr1e4_ss15_pi0 --resume \
  --fsdp-devices=4 \
  --batch-size=256 \
  --data.shuffle-buffer-size=200000 \
  --data.summation-steps=15 \
  --weight-loader CheckpointWeightLoader \
  --weight-loader.params-path gs://openpi-assets/checkpoints/pi0_base/params

  # --weight-loader PaliGemmaWeightLoader