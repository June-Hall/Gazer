#!/bin/bash
set -euo pipefail
set -x

cd "$(dirname "$0")" || exit 1

VLM_PORT=${VLM_PORT:-16660}
VLM_HOST=${VLM_HOST:-127.0.0.1}
VLM_STARTUP_TIMEOUT=${VLM_STARTUP_TIMEOUT:-1800}
KEEP_VLM_SERVER=${KEEP_VLM_SERVER:-0}

INFER_CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES:-0}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-../models/InfinityStar/}
OUTPUT_PATH=${OUTPUT_PATH:-./results/demo.mp4}
PROMPT=${PROMPT:-A girl is blowing bubbles in the yard, while apuppy is jumping}
SEED=${SEED:-42}
REFLECTION_START_SCALE=${REFLECTION_START_SCALE:-0}
REFLECTION_END_SCALE=${REFLECTION_END_SCALE:-1.0}
REFLECTION_INTERVAL=${REFLECTION_INTERVAL:-4}

mkdir -p logs

port_open() {
    nc -z "${VLM_HOST}" "${VLM_PORT}" >/dev/null 2>&1
}

VLM_SERVER_PID=""
STARTED_VLM_SERVER=0

cleanup() {
    if [[ "${STARTED_VLM_SERVER}" == "1" && "${KEEP_VLM_SERVER}" != "1" ]]; then
        kill "${VLM_SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if port_open; then
    echo "VLM server already running at http://${VLM_HOST}:${VLM_PORT}; reusing it."
else
    VLM_LOG="logs/vlm_server_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting VLM server at http://${VLM_HOST}:${VLM_PORT}; log: ${VLM_LOG}"

    # The server code currently uses device cuda:6 internally, so do not inherit
    # the inference CUDA_VISIBLE_DEVICES mask here.
    env -u CUDA_VISIBLE_DEVICES python vlm/video_evaluation_server.py > "${VLM_LOG}" 2>&1 &
    VLM_SERVER_PID=$!
    STARTED_VLM_SERVER=1

    elapsed=0
    while ! port_open; do
        if ! kill -0 "${VLM_SERVER_PID}" >/dev/null 2>&1; then
            echo "VLM server exited before becoming ready. Last log lines:"
            tail -n 80 "${VLM_LOG}" || true
            exit 1
        fi

        if (( elapsed >= VLM_STARTUP_TIMEOUT )); then
            echo "Timed out waiting for VLM server after ${VLM_STARTUP_TIMEOUT}s. Last log lines:"
            tail -n 80 "${VLM_LOG}" || true
            exit 1
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo "VLM server is ready at http://${VLM_HOST}:${VLM_PORT}"
fi

CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES}" python tools/infer_video_720p_demo.py \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --prompt "${PROMPT}" \
    --seed "${SEED}" \
    --reflection_start_scale "${REFLECTION_START_SCALE}" \
    --reflection_end_scale "${REFLECTION_END_SCALE}" \
    --reflection_interval "${REFLECTION_INTERVAL}" \
    "$@"
