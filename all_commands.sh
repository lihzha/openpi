curl -LsSf https://astral.sh/uv/install.sh | sh &&
source ~/.zshrc &&
GIT_LFS_SKIP_SMUDGE=1 uv sync &&
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . &&
source ~/.zshrc &&
# export WANDB_API_KEY=9d133998a3d44bf5dd2d827a5d8e2710dc91a19b &&
# export CURL_CA_BUNDLE=/home/irom-lab/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
# export SSL_CERT_FILE=/home/irom-lab/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
# export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/home/irom-lab/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
# export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# export OPENAI_DATA_HOME="gs://pi0-cot/cache"
uv run --group rlds scripts/compute_norm_stats.py --config-name pi0_droid_cot --max-frames 100_000 &&




