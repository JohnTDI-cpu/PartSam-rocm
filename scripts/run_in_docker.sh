#!/usr/bin/env bash
# Default entry inside the Docker container — runs PartSAM on /PartSAM/data_eval/
# Override defaults from outside via env, e.g.:
#   docker run ... -e BATCH=8 -e NUM_POINTS=50000 partsam-rocm
set -euo pipefail
cd /PartSAM
exec python evaluation/eval_everypart.py \
    eval_params.batch_size="${BATCH:-4}" \
    num_points="${NUM_POINTS:-10000}" \
    eval_params.use_graph_cut="${USE_GRAPH_CUT:-False}" \
    "$@"
