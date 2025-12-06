#!/bin/bash
set -euo pipefail


export CLOUDSDK_CORE_DISABLE_PROMPTS=1  # avoid interactive prompts

# ──────────────────────────────────────────────────────────────────────────────
# TPU Auto-Launcher with SSH hardening and timeout-based backoff
# ──────────────────────────────────────────────────────────────────────────────

# Load TPU helper functions explicitly (POSIX-compatible)
if ! command -v v6 >/dev/null 2>&1; then
  if [ -r "$HOME/.tpu_funcs.sh" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.tpu_funcs.sh"
  fi
fi

# -------- Timeouts (tunable) --------
TIMEOUT_BIN=${TIMEOUT_BIN:-timeout}
command -v "$TIMEOUT_BIN" >/dev/null 2>&1 || TIMEOUT_BIN=$(command -v gtimeout || echo timeout)

SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-12}     # seconds for initial TCP connect
SSH_ALIVE_INTERVAL=${SSH_ALIVE_INTERVAL:-10}       # keepalive ping every N sec
SSH_ALIVE_COUNT_MAX=${SSH_ALIVE_COUNT_MAX:-3}      # give up after N missed keepalives
SSH_TOTAL_TIMEOUT=${SSH_TOTAL_TIMEOUT:-60}         # total wall time per SSH
SSH_KILL_AFTER=${SSH_KILL_AFTER:-5}                # grace period before SIGKILL
DESCRIBE_TIMEOUT=${DESCRIBE_TIMEOUT:-20}           # gcloud describe timeout

# -------- Backoff (tunable) --------
SLEEP_SECS=${SLEEP_SECS:-20}

# -------- Helpers to standardize SSH flags & timeouts --------

# Build commonly used SSH flags so connections fail fast instead of hanging forever.
# Note: each value is passed as its own --ssh-flag argument for gcloud.
build_ssh_flag_args() {
  printf -- "--ssh-flag=%q " "-o BatchMode=yes"
  printf -- "--ssh-flag=%q " "-o ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
  printf -- "--ssh-flag=%q " "-o ServerAliveInterval=${SSH_ALIVE_INTERVAL}"
  printf -- "--ssh-flag=%q " "-o ServerAliveCountMax=${SSH_ALIVE_COUNT_MAX}"
  printf -- "--ssh-flag=%q " "-o StrictHostKeyChecking=accept-new"
  printf -- "--ssh-flag=%q " "-o UserKnownHostsFile=/dev/null"
}

# Run any command with a hard wall-clock timeout.
with_timeout() {
  local secs="$1"; shift
  # allow an optional "--" separator; ignore it if present
  [[ "${1:-}" == "--" ]] && shift
  if ! command -v "$TIMEOUT_BIN" >/dev/null 2>&1; then
    echo "$(ts) - ERROR: timeout binary '$TIMEOUT_BIN' not found"
    return 127
  fi
  "$TIMEOUT_BIN" -k "${SSH_KILL_AFTER}s" "${secs}s" "$@"
}


gc_ssh() {
  local tpu_name="$1"; shift

  # Optional dedicated key
  local key_arg=()
  if [[ -n "${GCLOUD_SSH_KEY_FILE:-}" && -r "${GCLOUD_SSH_KEY_FILE}" ]]; then
    key_arg=( "--ssh-key-file=${GCLOUD_SSH_KEY_FILE}" )
  fi

  # Pass raw ssh options AFTER a literal “--”
  local ssh_passthru=( "--"
    -o BatchMode=yes
    -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
    -o ServerAliveInterval="${SSH_ALIVE_INTERVAL}"
    -o ServerAliveCountMax="${SSH_ALIVE_COUNT_MAX}"
    -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile=/dev/null
  )

  local out rc
  out=$(with_timeout "$SSH_TOTAL_TIMEOUT" \
        gcloud alpha compute tpus tpu-vm ssh "$tpu_name" \
          --zone="$TPU_ZONE_v6" --project="$TPU_PROJECT" \
          "${key_arg[@]}" \
          "$@" \
          "${ssh_passthru[@]}" 2>&1)
  rc=$?

  if (( rc != 0 )); then
    # Show something useful (first matching error line, else the last line)
    local hint
    hint=$(printf "%s" "$out" | grep -Ei 'ERROR|invalid|unknown|unrecognized|usage|denied|refused' -m1 || true)
    echo "$(ts) - gc_ssh error: ${hint:-$(echo "$out" | tail -n1)}" >&2
    return $rc
  fi

  printf "%s" "$out"
}



# v6-style wrappers (your helpers ultimately call gcloud ssh).
# We cannot inject ssh flags into your v6 implementation, but the outer timeout protects us.
safe_v6() {
  local cmd="$1"
  with_timeout "$SSH_TOTAL_TIMEOUT" \
    bash -lc "source \"\$HOME/.tpu_funcs.sh\" 2>/dev/null || true; v6 $(printf '%q' "$cmd")"
}

safe_v6_tmux() {
  local cmd="$1"
  with_timeout "$SSH_TOTAL_TIMEOUT" \
    bash -lc "source \"\$HOME/.tpu_funcs.sh\" 2>/dev/null || true; v6_tmux $(printf '%q' "$cmd")"
}

# Log helper
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Returns 0 (busy) if any python proc has libtpu/XLA mapped; 1 (idle).
# On ANY SSH problem/timeout: treat as "busy" (return 0) so we don't launch and instead re-check state.
check_tpu_activity() {
  local tpu_name="$1"
  echo "$(ts) - Checking TPU activity (processes using libtpu)..."

  # ... your READY check stays the same ...

  # --- build probe locally; no quoting hazards here ---
  local probe enc
  probe=$(cat <<'PROBE'
set -e -o pipefail
uid=$(id -u)
for m in /proc/*/maps; do
  [ -r "$m" ] || continue
  if grep -qE 'libtpu|libxla|_xla_extension|libdevice' "$m" 2>/dev/null; then
    pid=${m%/maps}; pid=${pid#/proc/}
    puid=$(awk '/^Uid:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo -1)
    if [ "$puid" = "$uid" ]; then
      echo busy; exit 0
    fi
  fi
done
# conservative fallback: any python-like process
if pgrep -af '(^|/)python([0-9.])?' >/dev/null 2>&1; then
  echo busy; exit 0
fi
echo idle
PROBE
)
  enc=$(printf '%s' "$probe" | base64 | tr -d '\n')

  local out rc
  out=$(gc_ssh "$tpu_name" --worker=all \
        --command "bash -lc 'echo $enc | base64 -d | bash -s'" 2>&1)
  rc=$?

  if (( rc != 0 )); then
    echo "$(ts) - SSH probe failed (rc=$rc); treating as busy. Last line: $(echo "$out" | tail -n1)"
    return 0
  fi
  if printf '%s\n' "$out" | grep -Eq '(^|[\r\n])busy([\r\n]|$)'; then
    echo "$(ts) - Detected active TPU-bound processes."; return 0
  fi
  echo "$(ts) - No active TPU-bound processes detected; safe to launch."; return 1
}




usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  -f, --force              Force setup and training even if TPU is READY"
  echo "  -n, --tpu-num NUM        TPU chips: 8->2x2x2, 16->2x2x4, 32->2x4x4"
  echo "  -h, --help               Show this help message"
  echo ""
  exit 1
}

FORCE_RUN=false
TPU_NUM=${TPU_NUM:-8}
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--force) FORCE_RUN=true; shift ;;
    -n|--tpu-num)
      if [[ -n "${2:-}" ]]; then TPU_NUM="$2"; shift 2; else echo "Error: --tpu-num requires a value"; exit 1; fi ;;
    -h|--help)  usage ;;
    *)          EXTRA_ARGS+=("$1"); shift ;;
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

describe_tpu() {
  local out rc
  out=$(with_timeout "$DESCRIBE_TIMEOUT" -- \
        gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
          --zone="$TPU_ZONE_v6" \
          --project="$TPU_PROJECT" \
          --format="value(state)" 2>&1)
  rc=$?

  if (( rc == 0 )); then
    STATE="${out:-UNKNOWN}"
    return 0
  fi

  # Normalize common failure modes into states we can act on:
  if echo "$out" | grep -qiE 'not\s*found|404'; then
    STATE="NOT_FOUND"; return 0
  fi
  if echo "$out" | grep -qiE 'permission_denied|forbidden|permission.*denied|403'; then
    STATE="PERMISSION_DENIED"; return 0
  fi
  if echo "$out" | grep -qiE 'Invalid value for \[--zone\]|argument --zone'; then
    echo "$(ts) - ERROR: invalid or empty TPU_ZONE_v6='$TPU_ZONE_v6'"
    return 2
  fi

  echo "$(ts) - Describe error (rc=$rc): $(echo "$out" | tail -n1)"
  return 1
}

# Jittered sleep helper to avoid thundering herd/rate limits.
sleep_backoff() {
  local base="${1:-$SLEEP_SECS}"
  local jitter=$(( RANDOM % 7 ))
  echo "$(ts) - Next check in ${base}s (+${jitter}s jitter)..."
  sleep $(( base + jitter ))
}

# Clean shutdown
trap 'echo "$(ts) - Caught signal, exiting."; exit 0' INT TERM

while true; do
  echo "$(ts) - Checking TPU state..."
  if ! describe_tpu; then
    case $? in
      2) exit 1 ;;  # fatal misconfig (e.g., bad zone)
      *) sleep_backoff "$SLEEP_SECS"; continue ;;
    esac
  fi

  echo "$(ts) - TPU $TPU_NAME state: $STATE"

  run_setup_and_training=false

  case "$STATE" in
    NOT_FOUND|PREEMPTED|STOPPED)
      echo "$(ts) - Need to (re)create TPU..."
      # delete if present (guarded + timeout)
      if [[ "$STATE" != "NOT_FOUND" ]]; then
        if ! with_timeout $((DESCRIBE_TIMEOUT * 20)) -- \
              gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" \
                --zone="$TPU_ZONE_v6" --project="$TPU_PROJECT" --quiet; then
          echo "$(ts) - Delete failed/timed out."
          sleep_backoff "$SLEEP_SECS"; continue
        fi
      fi

      echo "$(ts) - Creating new TPU..."
      if ! with_timeout $((DESCRIBE_TIMEOUT * 20)) -- \
            gcloud alpha compute tpus tpu-vm create "$TPU_NAME" \
              --zone="$TPU_ZONE_v6" --project="$TPU_PROJECT" \
              --accelerator-type=v6e-${TPU_NUM} \
              --version=v2-alpha-tpuv6e \
              --service-account=irom-service-account@mae-irom-lab-guided-data.iam.gserviceaccount.com \
              --spot; then
        echo "$(ts) - Create failed/timed out."
        sleep_backoff "$SLEEP_SECS"; continue
      fi

      echo "$(ts) - Waiting for TPU to be READY..."
      sleep 10
      run_setup_and_training=true
      ;;

    PERMISSION_DENIED)
      echo "$(ts) - PERMISSION_DENIED from describe. Check IAM/API enablement."
      sleep_backoff "$SLEEP_SECS"
      continue
      ;;

    READY)
      # echo "$(ts) - TPU is READY. Probing activity..."
      # if check_tpu_activity "$TPU_NAME"; then
      #   echo "$(ts) - Busy or probe failed; will wait."
      #   run_setup_and_training=false
      # else
      #   echo "$(ts) - Idle; safe to launch."
      #   run_setup_and_training=true
      # fi
      if [[ "$FORCE_RUN" == "true" ]]; then
        run_setup_and_training=true
      else
        run_setup_and_training=false
      fi
      ;;

    *)
      echo "$(ts) - TPU in state: $STATE (not actionable now)."
      sleep_backoff "$SLEEP_SECS"
      continue
      ;;
  esac

  EXTRA_ARGS_STR=$( printf ' %q' "${EXTRA_ARGS[@]}" )

  if [[ "$run_setup_and_training" == "true" ]]; then
    echo "$(ts) - Setting up environment and repository..."
    if ! safe_v6 "curl -LsSf https://astral.sh/uv/install.sh | sh && \
        echo 'export WANDB_API_KEY=9d133998a3d44bf5dd2d827a5d8e2710dc91a19b' >> ~/.zshrc && \
        echo 'export OPENPI_DATA_HOME=\"$TPU_BUCKET/cache\"' >> ~/.zshrc && \
        source ~/.zshrc && \
        git clone --recurse-submodules https://github.com/lihzha/openpi-cot.git || true && \
        cd openpi-cot && \
        uv sync"; then
      echo "$(ts) - Setup failed/SSH timed out. Back to state check."
      sleep_backoff "$SLEEP_SECS"; continue
    fi

    echo "$(ts) - Starting training..."
    if ! safe_v6_tmux "source ~/.zshrc && cd openpi-cot && \
            git pull origin main && \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
            uv run --group tpu scripts/train.py pi_droid_cot_v6 $EXTRA_ARGS_STR \
    "; then
      echo "$(ts) - Launch failed/SSH timed out. Back to state check."
      sleep_backoff "$SLEEP_SECS"; continue
    fi

    echo "$(ts) - Training started successfully!"
    if [[ "$FORCE_RUN" == "true" ]]; then
      echo "$(ts) - Force run requested; exiting."
      exit 0
    fi
  fi

  sleep_backoff "$SLEEP_SECS"
done