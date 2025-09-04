# setup buckects
# create a buffer called v6_east1d
gsutil mb -l us-east1 gs://v6_east1d
# copy the data to the buffer
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 gs://v6_east1d/droid/
gsutil -m cp -r droid-language-actions gs://v6_east1d/droid-lang-actions
gsutil -m cp -r metadata gs://v6_east1d/metadata
gsutil -m cp -r assets gs://v6_east1d/assets


# setup tpu environment

v4 "curl -LsSf https://astral.sh/uv/install.sh | sh && echo 'export WANDB_API_KEY=9d133998a3d44bf5dd2d827a5d8e2710dc91a19b' >> ~/.zshrc && echo 'export OPENPI_DATA_HOME="gs://pi0-cot/cache"' >> ~/.zshrc && source ~/.zshrc && git clone --branch tpu https://github.com/lihzha/openpi.git && cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . "
v6 "curl -LsSf https://astral.sh/uv/install.sh | sh && echo 'export WANDB_API_KEY=9d133998a3d44bf5dd2d827a5d8e2710dc91a19b' >> ~/.zshrc && echo 'export OPENPI_DATA_HOME="gs://v6_east1d/cache"' >> ~/.zshrc && source ~/.zshrc && git clone --branch tpu https://github.com/lihzha/openpi.git && cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . "


# train
v4 "source ~/.zshrc && cd openpi && git pull origin tpu && XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group rlds scripts/train.py pi0_droid_cot_v4 --exp-name=v4_fsdp4_bs256 --overwrite"
v6 "source ~/.zshrc && cd openpi && git pull origin tpu && XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group rlds scripts/train.py pi0_droid_cot_v6 --exp-name=v6_fsdp4_bs256 --resume"

v4_one "source ~/.zshrc && cd openpi && \
            git pull origin tpu && \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_BOUNDS=1,1,1 \       
            uv run --group rlds scripts/train.py pi0_droid_cot_v4 --fsdp-devices=8 --batch-size=64 --exp-name v4_bs256_lr1e4_ss15_pi0_max110_overfit150  --data.summation-steps=15 --data.max_samples=150 --data.sum-decimal=2f  --weight-loader.kind=checkpoint --weight-loader.params-path=gs://openpi-assets/checkpoints/pi0_base/params --save_interval=300 --log-interval=100 --data.left-pad --overwrite"

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
  --weight-loader=CheckpointWeightLoader \
  --weight-loader.params-path=gs://openpi-assets/checkpoints/pi0_base/params

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group rlds scripts/train.py pi0_droid_cot_v4 \
  --exp-name=v4_bs256_lr1e4_ss15_paligemma --resume \
  --fsdp-devices=4 \
  --batch-size=256 \
  --data.shuffle-buffer-size=200000 \
  --data.summation-steps=15 \
  --weight-loader PaliGemmaWeightLoader

v4 "source ~/.zshrc && cd openpi && git pull origin tpu && XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --group rlds scripts/train.py pi0_droid_cot_v4 --exp-name=v4_bs256_lr1e4_ss15_paligemma_max115 --resume --fsdp-devices=4 --batch-size=256 --data.summation-steps=15"


--weight-loader.params-path=gs://openpi-assets/checkpoints/pi0_base/params

#1. create tmux
tmux new -s pi0-cot
source ~/.tpu_env.sh
export TPU_NAME=pi0-cot
./watch_and_run.sh --exp-name v6_bs256_lr1e4_ss15_paligemma_max110 --fsdp-devices=8 --batch-size=256 --data.summation-steps=15 --weight-loader.kind=paligemma --resume
./watch_and_run.sh --exp-name v6_bs256_lr1e4_ss15_pi0_max110 --fsdp-devices=8 --batch-size=256 --data.summation-steps=15 --resume
./watch_and_run.sh --exp-name v6_bs256_lr1e4_ss15_pi0_max110_overfit150 --fsdp-devices=8 --batch-size=256 --data.summation-steps=15 --weight-loader.kind=checkpoint --weight-loader.params-path=gs://openpi-assets/checkpoints/pi0_base/params --data.max_samples=150 --resume


# 8 chips per process (2 hosts × 4 chips each)
alias TPU0_7="TPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=2,1,1 TPU_MESH_CONTROLLER_ADDRESS=10.130.0.2:8476 TPU_MESH_CONTROLLER_PORT=8476"
alias TPU8_15="TPU_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=2,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8477 TPU_MESH_CONTROLLER_PORT=8477"
alias TPU16_23="TPU_VISIBLE_DEVICES=16,17,18,19,20,21,22,23 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=2,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"
alias TPU24_31="TPU_VISIBLE_DEVICES=24,25,26,27,28,29,30,31 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=2,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8479 TPU_MESH_CONTROLLER_PORT=8479"
