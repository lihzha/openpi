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
