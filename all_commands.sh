curl -LsSf https://astral.sh/uv/install.sh | sh &&
export WANDB_API_KEY=9d133998a3d44bf5dd2d827a5d8e2710dc91a19b &&
source ~/.zshrc &&
GIT_LFS_SKIP_SMUDGE=1 uv sync &&
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .