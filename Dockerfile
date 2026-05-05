FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/lukeford22/BindBot.git"
LABEL org.opencontainers.image.description="BindBot Base Environment - Clone repo at runtime"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      git \
      rsync \
      libgfortran5 \
      tmux \
      wget \
      build-essential \
      pkg-config \
      procps \
      unzip \
      openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Configure SSH for RunPod (RunPod handles key injection automatically)
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    ssh-keygen -A

# Configure SSH daemon for RunPod
RUN echo "Port 22" >> /etc/ssh/sshd_config && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "AuthorizedKeysFile /root/.ssh/authorized_keys" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "ChallengeResponseAuthentication no" >> /etc/ssh/sshd_config && \
    echo "UsePAM yes" >> /etc/ssh/sshd_config && \
    echo "X11Forwarding yes" >> /etc/ssh/sshd_config && \
    echo "PrintMotd no" >> /etc/ssh/sshd_config && \
    echo "AcceptEnv LANG LC_*" >> /etc/ssh/sshd_config && \
    echo "Subsystem sftp /usr/lib/openssh/sftp-server" >> /etc/ssh/sshd_config

# Expose SSH port
EXPOSE 22

# Install OpenCL ICD loader and tools; register NVIDIA OpenCL ICD
RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 clinfo && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install Miniforge (Conda) at /miniforge3
ENV CONDA_DIR=/miniforge3
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh

# Put conda on PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Improve conda robustness and cleanup
RUN conda config --set channel_priority strict && \
    conda config --set always_yes yes && \
    conda update -n base -c conda-forge conda && \
    conda clean -afy

# ============================================
# INSTALL DEPENDENCIES TO /data (PERSISTENT)
# ============================================

# Create persistent data directory and copy minimal files needed for installation
RUN mkdir -p /data
WORKDIR /data

# Copy ONLY the installer script and functions directory
COPY install.sh /data/install.sh
COPY functions /data/functions

# Run the installer script from /data directory
# This will create: /data/params (weights) and make /data/functions binaries executable
ARG WITH_PYROSETTA=false
ENV WITH_PYROSETTA=${WITH_PYROSETTA}

RUN bash -lc 'set -e && \
    source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    cd /data && \
    EXTRA=""; if [ "${WITH_PYROSETTA}" != "true" ]; then EXTRA="--no-pyrosetta"; fi && \
    bash /data/install.sh --pkg_manager conda --cuda 12.1 ${EXTRA}'

# Verify installation succeeded
RUN test -f /data/params/params_model_5_ptm.npz || { echo "AlphaFold weights not found!"; exit 1; }
RUN test -x /data/functions/dssp || { echo "DSSP binary not executable!"; exit 1; }

# Remove installer script (no longer needed, saves space)
RUN rm -f /data/install.sh

# ============================================
# ENVIRONMENT VARIABLES
# ============================================

# Set PATH for BindCraft environment
ENV PATH=${CONDA_DIR}/envs/BindCraft/bin:${CONDA_DIR}/bin:${PATH} \
    LD_LIBRARY_PATH=${CONDA_DIR}/envs/BindCraft/lib:${LD_LIBRARY_PATH} \
    PYTHONUNBUFFERED=1

# Point BindCraft to persistent data locations
ENV BINDCRAFT_HOME=/app \
    BINDCRAFT_PARAMS=/data/params \
    BINDCRAFT_FUNCTIONS=/data/functions

# Prefer OpenCL (fallback to CUDA) in OpenMM by default
ENV OPENMM_PLATFORM_ORDER=OpenCL,CUDA \
    OPENMM_DEFAULT_PLATFORM=OpenCL

# GitHub repo to clone
ENV GITHUB_REPO=https://github.com/lukeford22/BindBot.git \
    GITHUB_BRANCH=main

# ============================================
# ENTRYPOINT - CLONE REPO AT RUNTIME
# ============================================

# Create empty /app directory (will be populated at runtime)
WORKDIR /app

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/bindbot-entrypoint.sh
RUN chmod +x /usr/local/bin/bindbot-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/bindbot-entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
