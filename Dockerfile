FROM isaaclab_ready:v1

USER root

SHELL ["/bin/bash", "-c"]

# ============================================================
# Paths
# ============================================================

ENV PYTHON_BIN=/home/user/isaaclab/bin/python
ENV ISAAC_SRC=/home/user/isaaclab/lib/python3.11/site-packages/isaaclab/source

# Project source will be MOUNTED here at runtime.
ENV VQR_ROOT=/workspace/vqr

# Offline Isaac asset root.
ENV ISAAC_ASSET_ROOT=/opt/isaac_assets/Assets/Isaac/5.1

ENV PATH="/home/user/isaaclab/bin:${PATH}"

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ALLOW_ROOT=1
ENV PYTHONUNBUFFERED=1

# Make pip strictly offline.
ENV PIP_NO_INDEX=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1


# ============================================================
# Verify base environment
# ============================================================

RUN ${PYTHON_BIN} - <<'PY'
import sys
import torch
import isaaclab
import gymnasium
import rsl_rl

print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("IsaacLab:", isaaclab.__file__)
print("Gymnasium:", gymnasium.__version__)
print("RSL-RL:", rsl_rl.__file__)
print("BASE ENV: PASS")
PY


# ============================================================
# Install Isaac Lab extensions from LOCAL source
# No Internet required.
# ============================================================

RUN test -d ${ISAAC_SRC}/isaaclab_tasks && \
    test -d ${ISAAC_SRC}/isaaclab_rl && \
    test -d ${ISAAC_SRC}/isaaclab_assets

RUN ${PYTHON_BIN} -m pip install \
    --no-deps \
    --no-build-isolation \
    -e ${ISAAC_SRC}/isaaclab_tasks

RUN ${PYTHON_BIN} -m pip install \
    --no-deps \
    --no-build-isolation \
    -e ${ISAAC_SRC}/isaaclab_rl

RUN ${PYTHON_BIN} -m pip install \
    --no-deps \
    --no-build-isolation \
    -e ${ISAAC_SRC}/isaaclab_assets


# ============================================================
# Offline assets
#
# Host build context (repository root):
#   Isaac/
#
# Container:
#   /opt/isaac_assets/Assets/Isaac/5.1/Isaac/
# ============================================================

COPY Isaac/ /opt/isaac_assets/Assets/Isaac/5.1/Isaac/


# ============================================================
# Verify offline assets
# ============================================================

RUN test -f \
    ${ISAAC_ASSET_ROOT}/Isaac/Environments/Grid/default_environment.usd && \
    test -f \
    ${ISAAC_ASSET_ROOT}/Isaac/Environments/Grid/Materials/Textures/Wireframe_blue.png && \
    test -f \
    ${ISAAC_ASSET_ROOT}/Isaac/Environments/Grid/Materials/Textures/WireframeBlur_basecolor.png && \
    test -f \
    ${ISAAC_ASSET_ROOT}/Isaac/Environments/Grid/Materials/Textures/WireframeBlur_blue.png && \
    test -f \
    ${ISAAC_ASSET_ROOT}/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr


# ============================================================
# Verify Isaac Lab packages
# ============================================================

RUN ${PYTHON_BIN} - <<'PY'
import isaaclab
import isaaclab_tasks
import isaaclab_rl
import isaaclab_assets
import torch
import rsl_rl

print("isaaclab       :", isaaclab.__file__)
print("isaaclab_tasks :", isaaclab_tasks.__file__)
print("isaaclab_rl    :", isaaclab_rl.__file__)
print("isaaclab_assets:", isaaclab_assets.__file__)
print("torch          :", torch.__version__)
print("rsl_rl         :", rsl_rl.__file__)

print("ISAAC ENV: PASS")
PY


# ============================================================
# Runtime bootstrap
#
# Project does NOT exist while building the image.
# At container startup it will be mounted at /workspace/vqr.
# Then install rl_training editable.
# ============================================================

RUN cat > /usr/local/bin/vqr-entrypoint <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT="/workspace/vqr"
PYTHON="/home/user/isaaclab/bin/python"

if [ ! -d "${PROJECT}" ]; then
    echo "[ERROR] Project volume not mounted:"
    echo "        ${PROJECT}"
    exit 1
fi

if [ ! -d "${PROJECT}/source/rl_training" ]; then
    echo "[ERROR] Cannot find:"
    echo "        ${PROJECT}/source/rl_training"
    echo
    echo "Check your docker -v source path."
    exit 1
fi

echo "[INFO] Project source: ${PROJECT}"

# Editable install.
# Python source changes are immediately visible without rebuilding image.
"${PYTHON}" -m pip install \
    --no-deps \
    --no-build-isolation \
    -e "${PROJECT}/source/rl_training" \
    >/tmp/vqr_pip_install.log 2>&1 || {
        cat /tmp/vqr_pip_install.log
        exit 1
    }

cd "${PROJECT}"

exec "$@"
EOF

RUN chmod +x /usr/local/bin/vqr-entrypoint


# ============================================================
# Training wrapper
#
# Forces Isaac asset root to LOCAL offline assets.
# ============================================================

RUN cat > /usr/local/bin/vqr-train <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT="/workspace/vqr"
PYTHON="/home/user/isaaclab/bin/python"
ASSETS="/opt/isaac_assets/Assets/Isaac/5.1"

cd "${PROJECT}"

exec "${PYTHON}" \
    scripts/reinforcement_learning/rsl_rl/train.py \
    "$@" \
    --kit_args="--/persistent/isaac/asset_root/default=${ASSETS} --/persistent/isaac/asset_root/cloud=${ASSETS} --/persistent/isaac/asset_root/nvidia=${ASSETS}"
EOF

RUN chmod +x /usr/local/bin/vqr-train


# ============================================================
# Final setup
# ============================================================

RUN mkdir -p /workspace/vqr

RUN chown -R user:user \
    /workspace \
    /opt/isaac_assets

USER user

WORKDIR /workspace/vqr

ENTRYPOINT ["/usr/local/bin/vqr-entrypoint"]

CMD ["/bin/bash"]
