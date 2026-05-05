# PartSAM on AMD ROCm — Docker image
#
# Builds once (~30 min, ~12 GB image), runs on demand:
#   docker build -t partsam-rocm .
#   docker run --rm -it \
#       --device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined \
#       --ipc=host --shm-size=8g \
#       -v $(pwd)/data_eval:/PartSAM/data_eval \
#       -v $(pwd)/results:/PartSAM/results \
#       partsam-rocm
#
# The image is dormant on disk between runs. Nothing auto-mounts on boot.

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_XNACK=1
# MIOpen flags — required to dodge the stoul crash + winograd hangs on gfx1201
ENV MIOPEN_DEBUG_CONV_GEMM=1
ENV MIOPEN_DEBUG_CONV_DIRECT=0
ENV MIOPEN_DEBUG_CONV_WINOGRAD=0
ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
ENV MIOPEN_USER_DB_PATH=/tmp/miopen_db
ENV MIOPEN_SYSTEM_DB_PATH=/tmp/miopen_db
ENV PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

WORKDIR /build

# --- 1. Install our exact torch + standard pip deps ---
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/rocm7.2 && \
    pip install --no-cache-dir lightning==2.2 h5py yacs trimesh scikit-image \
        loguru boto3 mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope \
        potpourri3d simple_parsing arrgh open3d safetensors \
        hydra-core omegaconf accelerate timm igraph ninja vtk huggingface_hub

# --- 2. Native CUDA→HIP extensions, FORCE_CUDA=1 trick ---
RUN git clone https://github.com/Mateusz-Dera/pytorch_cluster_rocm.git && \
    cd pytorch_cluster_rocm && pip install . --no-build-isolation && cd /build

RUN git clone --depth 1 https://github.com/rusty1s/pytorch_scatter.git && \
    cd pytorch_scatter && FORCE_CUDA=1 pip install . --no-build-isolation && cd /build

RUN git clone --depth 1 https://github.com/Pointcept/SAMPart3D.git && \
    cd SAMPart3D/libs/pointops && FORCE_CUDA=1 pip install . --no-build-isolation && cd /build

RUN git clone --depth 1 https://github.com/zyc00/Point-SAM.git && \
    cd Point-SAM && git submodule update --init third_party/torkit3d && \
    FORCE_CUDA=1 pip install third_party/torkit3d --no-build-isolation && cd /build

RUN git clone --depth 1 https://github.com/ROCm/apex.git && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" . && cd /build

# --- 3. Sanity check: all 5 must import ---
RUN python -c "import torch_cluster, torch_scatter, pointops, torkit3d, apex; print('all 5 native exts OK')"

# --- 4. Clone PartSAM upstream + download weights ---
RUN git clone --depth 1 https://github.com/czvvd/PartSAM.git /PartSAM && \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('Czvvd/PartSAM', local_dir='/PartSAM/pretrained')"

# --- 5. Apply ROCm patch + inject into eval entry ---
COPY patches/partsam_patch.py /PartSAM/rocm_patch/partsam_patch.py
RUN python -c "\
fn='/PartSAM/evaluation/eval_everypart.py';\
src=open(fn).read();\
new='import sys\nsys.path.insert(0,\"/PartSAM/rocm_patch\")\nimport partsam_patch\n'+src;\
open(fn,'w').write(new) if 'partsam_patch' not in src else None"

# --- 6. Pre-create dirs + drop run script ---
RUN mkdir -p /PartSAM/results /tmp/miopen_db
WORKDIR /PartSAM
COPY scripts/run_in_docker.sh /usr/local/bin/run-partsam
RUN chmod +x /usr/local/bin/run-partsam

ENTRYPOINT ["/usr/local/bin/run-partsam"]
CMD []
