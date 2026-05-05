#!/usr/bin/env bash
# install.sh — one-shot PartSAM ROCm installer for AMD RDNA3/RDNA4 GPUs.
# Verified on AMD Radeon AI PRO R9700 (gfx1201) with ROCm 7.2 + PyTorch 2.11.
#
# Usage:
#   ./install.sh                    # full install to ~/partsam_venv + /tmp/PartSAM
#   VENV=~/myvenv ./install.sh      # custom venv path
#   PARTSAM_DIR=~/PartSAM ./install.sh  # custom checkout path

set -euo pipefail

VENV="${VENV:-$HOME/partsam_venv}"
PARTSAM_DIR="${PARTSAM_DIR:-/tmp/PartSAM}"
BUILD_DIR="${BUILD_DIR:-/tmp/partsam_build}"
PATCH_SRC="$(cd "$(dirname "$0")/../patches" && pwd)/partsam_patch.py"

echo "=== PartSAM ROCm installer ==="
echo "venv: $VENV"
echo "partsam: $PARTSAM_DIR"
echo "build cache: $BUILD_DIR"

# 1. Verify ROCm + Python
command -v rocminfo >/dev/null || { echo "ERROR: rocminfo not found. Install ROCm 7.x first."; exit 1; }
# NOTE: split pipeline — `rocminfo | grep -m1` triggers SIGPIPE under `set -euo pipefail`
# (grep closes stdin after first match → rocminfo dies → pipefail aborts script with exit 141).
ROCMINFO_OUT=$(rocminfo 2>/dev/null || true)
GFX=$(echo "$ROCMINFO_OUT" | grep -m1 'Name:.*gfx' | awk '{print $2}')
echo "GPU: $GFX"
case "$GFX" in
  gfx1100|gfx1101|gfx1102|gfx1200|gfx1201) echo "OK — RDNA3/RDNA4 supported" ;;
  *) echo "WARN: $GFX untested. Continuing." ;;
esac
command -v python3 >/dev/null || { echo "ERROR: python3 missing"; exit 1; }

# 2. venv + torch ROCm 7.2
if [ ! -d "$VENV" ]; then
  echo "[1/7] Creating venv..."
  python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
# Pin setuptools <82 — torch 2.11+rocm7.2 wheels are pinned to setuptools<82,
# newer versions trigger an installer warning and can break editable extension builds.
$PIP install -q --upgrade pip wheel "setuptools<82"
echo "[2/7] Installing torch + torchvision (ROCm 7.2)..."
$PIP install -q torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('  torch ' + torch.__version__ + ' on ' + torch.cuda.get_device_name(0))"

# 3. Standard pip dependencies
echo "[3/7] Installing standard pip deps..."
$PIP install -q lightning==2.2 h5py yacs trimesh scikit-image loguru boto3 \
                mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope \
                potpourri3d simple_parsing arrgh open3d safetensors \
                hydra-core omegaconf accelerate timm igraph ninja vtk huggingface_hub

# 4. Build 5 native CUDA extensions on ROCm via FORCE_CUDA=1 trick
echo "[4/7] Building native extensions on ROCm (FORCE_CUDA=1 trick)..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

if ! $PY -c "import torch_cluster" 2>/dev/null; then
  [ -d pytorch_cluster_rocm ] || git clone https://github.com/Mateusz-Dera/pytorch_cluster_rocm.git
  (cd pytorch_cluster_rocm && $PIP install . --no-build-isolation)
fi

if ! $PY -c "import torch_scatter" 2>/dev/null; then
  [ -d pytorch_scatter ] || git clone --depth 1 https://github.com/rusty1s/pytorch_scatter.git
  (cd pytorch_scatter && FORCE_CUDA=1 $PIP install . --no-build-isolation)
fi

if ! $PY -c "import pointops" 2>/dev/null; then
  [ -d SAMPart3D ] || git clone --depth 1 https://github.com/Pointcept/SAMPart3D.git
  (cd SAMPart3D/libs/pointops && FORCE_CUDA=1 $PIP install . --no-build-isolation)
fi

if ! $PY -c "import torkit3d" 2>/dev/null; then
  [ -d Point-SAM ] || git clone --depth 1 https://github.com/zyc00/Point-SAM.git
  (cd Point-SAM && git submodule update --init third_party/torkit3d && \
   FORCE_CUDA=1 $PIP install third_party/torkit3d --no-build-isolation)
fi

if ! $PY -c "import apex" 2>/dev/null; then
  # ROCm/apex repo uses relative symlinks under apex/contrib/csrc.
  # exFAT/FAT32/some NTFS-via-fuse mounts cannot store POSIX symlinks — git stores them as
  # plain text files containing the target path, which apex's hipify_extension walks into
  # as a directory and crashes with `NotADirectoryError`. Probe and fall back to ext4 HOME.
  APEX_BUILD_DIR="$BUILD_DIR/apex"
  if ! ln -sf "$BUILD_DIR" "$BUILD_DIR/.symlink_test" 2>/dev/null || [ ! -L "$BUILD_DIR/.symlink_test" ]; then
    echo "  WARN: $BUILD_DIR cannot store POSIX symlinks (likely exFAT/FAT32/non-posix FS)."
    echo "  WARN: cloning apex into ~/.partsam_apex_build instead (ext4/xfs assumed)."
    APEX_BUILD_DIR="$HOME/.partsam_apex_build/apex"
    mkdir -p "$HOME/.partsam_apex_build"
  fi
  rm -f "$BUILD_DIR/.symlink_test"
  [ -d "$APEX_BUILD_DIR" ] || git clone --depth 1 https://github.com/ROCm/apex.git "$APEX_BUILD_DIR"
  (cd "$APEX_BUILD_DIR" && $PIP install -v --disable-pip-version-check --no-cache-dir \
      --no-build-isolation --config-settings "--build-option=--cpp_ext" .)
fi

$PY -c "import torch_cluster, torch_scatter, pointops, torkit3d, apex; print('  ALL 5 native extensions OK')"

# 5. Clone PartSAM + download weights
echo "[5/7] Cloning PartSAM + downloading weights..."
[ -d "$PARTSAM_DIR" ] || git clone --depth 1 https://github.com/czvvd/PartSAM.git "$PARTSAM_DIR"
[ -d "$PARTSAM_DIR/pretrained" ] || \
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Czvvd/PartSAM', local_dir='$PARTSAM_DIR/pretrained')"

# 6. Apply ROCm patch
echo "[6/7] Applying ROCm patch (apex + LayerNorm fallback)..."
mkdir -p "$PARTSAM_DIR/rocm_patch" && cp "$PATCH_SRC" "$PARTSAM_DIR/rocm_patch/partsam_patch.py"
$PY <<EOF
fn = '$PARTSAM_DIR/evaluation/eval_everypart.py'
src = open(fn).read()
if 'partsam_patch' not in src:
    new = 'import sys\nsys.path.insert(0, "$PARTSAM_DIR/rocm_patch")\nimport partsam_patch  # ROCm safety stubs\n' + src
    open(fn, 'w').write(new)
    print('  patched evaluation/eval_everypart.py')
else:
    print('  already patched')
EOF

# 7. Create results dir + write run.sh wrapper
mkdir -p "$PARTSAM_DIR/results"
cat > "$PARTSAM_DIR/run_rocm.sh" <<'RUN_EOF'
#!/usr/bin/env bash
# Run PartSAM on AMD ROCm with all required env flags + reduced batch for 16GB cards.
cd "$(dirname "$0")"
export HSA_XNACK=1
export MIOPEN_DEBUG_CONV_GEMM=1
export MIOPEN_DEBUG_CONV_DIRECT=0
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export MIOPEN_USER_DB_PATH=/tmp/miopen_db
export MIOPEN_SYSTEM_DB_PATH=/tmp/miopen_db
mkdir -p /tmp/miopen_db
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

VENV="${VENV:-$HOME/partsam_venv}"
exec "$VENV/bin/python" evaluation/eval_everypart.py \
  eval_params.batch_size="${BATCH:-4}" \
  num_points="${NUM_POINTS:-10000}" \
  eval_params.use_graph_cut="${USE_GRAPH_CUT:-False}" \
  "$@"
RUN_EOF
chmod +x "$PARTSAM_DIR/run_rocm.sh"

echo "[7/7] DONE."
echo
echo "=== Install complete ==="
echo "Run inference:"
echo "  cd $PARTSAM_DIR"
echo "  ./run_rocm.sh                                    # all meshes in data_eval/"
echo
echo "  USE_GRAPH_CUT=True ./run_rocm.sh                 # sharper part boundaries (only for meshes <50k faces)"
echo "  BATCH=8 NUM_POINTS=50000 ./run_rocm.sh           # higher quality (needs 32GB VRAM)"
