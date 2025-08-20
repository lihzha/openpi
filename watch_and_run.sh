#!/bin/bash
set -euo pipefail

# TPU Auto-Launcher with Memory Usage Monitoring
# This script automatically manages TPU creation, setup, and training launch.
# It includes logic to check if TPU memory usage is above 20% before launching
# new training programs to avoid conflicts with existing JAX processes.

# Configuration

# Load TPU helper functions explicitly (POSIX-compatible)
if ! command -v v6 >/dev/null 2>&1; then
  if [ -r "$HOME/.tpu_funcs.sh" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.tpu_funcs.sh"
  fi
fi

# Returns 0 (busy) if any python process has libtpu/XLA mapped; 1 (idle) otherwise.
check_tpu_activity() {
  local tpu_name="$1"

  echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking TPU activity (processes using libtpu)..."

  # Verify TPU is READY first
  local state
  state=$(gcloud alpha compute tpus tpu-vm describe "$tpu_name" \
            --zone="$TPU_ZONE_v6" --project="$TPU_PROJECT" \
            --format="value(state)" 2>/dev/null || echo "")
  if [[ "$state" != "READY" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU not READY (state=$state). Treating as idle for launcher gating."
    return 1
  fi

  # Run on ALL workers so a single busy worker marks TPU as busy
  # If your v6 helper already targets all workers, you can keep v6; otherwise use gcloud directly.
  local busy
  busy=$(gcloud alpha compute tpus tpu-vm ssh "$tpu_name" \
           --zone="$TPU_ZONE_v6" --project="$TPU_PROJECT" \
           --worker=all \
           --command="bash -lc '
             set -euo pipefail
             PIDS=\$(pgrep -u \$(id -u) -x python 2>/dev/null || true)
             # Fall back: also catch python3
             PIDS=\"\$PIDS \$(pgrep -u \$(id -u) -x python3 2>/dev/null || true)\"
             if [[ -z \"\$PIDS\" ]]; then
               echo idle
               exit 0
             fi
             for pid in \$PIDS; do
               # Check if the process has TPU/XLA libs mapped
               if grep -E \"libtpu|libxla|_xla_extension|libdevice\" \"/proc/\$pid/maps\" >/dev/null 2>&1; then
                 echo busy
                 exit 0
               fi
             done
             echo idle
           '" 2>/dev/null | tr -d '\r' | tail -n1)

  if [[ "$busy" == "busy" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Detected active Python processes using libtpu/XLA."
    return 0  # busy
  fi

  # Optional extra heuristic: active TPU runtime logs in the last minute
  local recent_logs
  recent_logs=$(gcloud alpha compute tpus tpu-vm ssh "$tpu_name" \
                  --zone="$TPU_ZONE_v6" --project="$TPU_PROJECT" \
                  --worker=all \
                  --command="bash -lc '
                    journalctl -n 100 --since \"1 min ago\" 2>/dev/null | grep -Ei \"libtpu|xla|tpu runtime\" || true
                  '" 2>/dev/null)
  if [[ -n "$recent_logs" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Recent TPU/XLA runtime logs detected; treating as busy."
    return 0
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S') - No active TPU-bound processes detected; safe to launch."
  return 1  # idle
}


usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -f, --force              Force setup and training even if TPU is READY"
    echo "  -h, --help               Show this help message"
    echo ""
    exit 1
}

FORCE_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required parameters
if [[ -z "${TPU_NAME:-}" ]]; then
    echo "Error: TPU name cannot be empty"
    exit 1
fi

if ! command -v v6 >/dev/null 2>&1; then
  echo "ERROR: TPU helper functions (e.g., v6) not found."
  echo "Hint: move them into ~/.tpu_funcs.sh and source from your rc files (e.g. ~/.bashrc, ~/.zshrc)."
  exit 1
fi

echo "Starting TPU auto-launcher with:"
echo "  TPU Name: ${TPU_NAME:-}"
echo "  Zone: $TPU_ZONE_v6"
echo "  Project: $TPU_PROJECT"
echo "  Bucket: $TPU_BUCKET"
echo "  Force run: $FORCE_RUN"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args: ${EXTRA_ARGS[*]}"
fi
echo ""


while true; do
  # Check TPU state
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking TPU state..."
  STATE=$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$TPU_ZONE_v6" \
    --project="$TPU_PROJECT" \
    --format="value(state)" 2>/dev/null || echo "NOT_FOUND")

  echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU $TPU_NAME state: $STATE"

  if [[ "$STATE" == "PREEMPTED" || "$STATE" == "STOPPED" || "$STATE" == "NOT_FOUND" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU $TPU_NAME is $STATE. Re-creating..."

    # Delete if exists in PREEMPTED state
    if [[ "$STATE" != "NOT_FOUND" ]]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Deleting existing TPU..."
      gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" \
        --zone="$TPU_ZONE_v6" \
        --project="$TPU_PROJECT" \
        --quiet || true
    fi

    # ---------------------------
    # Command 1: Create TPU
    # ---------------------------
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Creating new TPU..."
    if ! gcloud alpha compute tpus tpu-vm create "$TPU_NAME" \
      --zone="$TPU_ZONE_v6" \
      --project="$TPU_PROJECT" \
      --accelerator-type=v6e-8 \
      --version=v2-alpha-tpuv6e \
      --service-account=irom-service-account@mae-irom-lab-guided-data.iam.gserviceaccount.com \
      --spot; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed to create TPU. Retrying in 20 seconds..."
      sleep 20
      continue
    fi

    # Wait a bit for TPU to be ready
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Waiting for TPU to be ready..."
    sleep 30

    # Run setup and training
    run_setup_and_training=true
  elif [[ "$STATE" == "READY" && "$FORCE_RUN" == "true" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU is READY but force flag is set. Running setup and training..."
    run_setup_and_training=true
  elif [[ "$STATE" == "READY" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU is ready and running"
    # if check_tpu_activity "$TPU_NAME"; then
    #   echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU appears busy. Waiting..."
    #   run_setup_and_training=false
    # else
    #   echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU appears idle. Safe to launch."
    #   run_setup_and_training=true
    # fi
    # run_setup_and_training=true
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - TPU in state: $STATE"
    run_setup_and_training=false
  fi

  if [[ "$run_setup_and_training" == "true" ]]; then
    # ---------------------------
    # Command 2: Env + Repo Setup
    # ---------------------------
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Setting up environment and repository..."
    if ! v6 "curl -LsSf https://astral.sh/uv/install.sh | sh && \
        echo 'export WANDB_API_KEY=9d133998a3d44bf5dd2d827a5d8e2710dc91a19b' >> ~/.zshrc && \
        echo 'export OPENPI_DATA_HOME=\"$TPU_BUCKET/cache\"' >> ~/.zshrc && \
        source ~/.zshrc && \
        git clone --branch tpu https://github.com/lihzha/openpi.git || true && \
        cd openpi && \
        GIT_LFS_SKIP_SMUDGE=1 uv sync && \
        GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ."; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed to setup environment. Retrying in 20 seconds..."
      sleep 20
      continue
    fi

    # ---------------------------
    # Command 3: Start Training
    # ---------------------------
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting training"
    if ! v6_tmux "source ~/.zshrc && cd openpi && \
            git pull origin tpu && \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
            uv run --group rlds scripts/train.py pi0_droid_cot_v6 \
            $( printf ' %q' "${EXTRA_ARGS[@]}" )
    "; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed to start training. Retrying in 20 seconds..."
      sleep 20    
      continue
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Training started successfully!"
    
    # If force run was used, exit after one iteration instead of continuous monitoring
    if [[ "$FORCE_RUN" == "true" ]]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Force run completed. Exiting..."
      exit 0
    fi
  fi

  # Poll every 60s
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Next check in 20 seconds..."
  sleep 20
done
