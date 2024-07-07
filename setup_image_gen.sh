#!/bin/bash
# BUILD_DIR="/home/ubuntu"  # T4
BUILD_DIR="/mnt/resource_nvme"  # A100

NVIDIA_DRIVER_BRANCH="550"

# Must use nightly to compile model on python 3.12
MINICONDA3_PYTHON_MAJOR_VERSION="3"
MINICONDA3_PYTHON_MINOR_VERSION="11"
MINICONDA3_VERSION="24.5.0-0"
MINICONDA3_INSTALLER_FILE_NAME="Miniconda3-py${MINICONDA3_PYTHON_MAJOR_VERSION}${MINICONDA3_PYTHON_MINOR_VERSION}_${MINICONDA3_VERSION}-Linux-x86_64.sh"
MINICONDA3_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/${MINICONDA3_INSTALLER_FILE_NAME}"

download () {
    local save_path=$1
    local url=$2
    curl -fsSL -o "${save_path}" "${url}"
}

# Install NVIDIA driver
# NVIDIA open driver is production ready for NVIDIA Turing and NVIDIA Ampere architecture families
# NC A100 v4-series -> NVIDIA A100 -> Ampere
# NCasT4_v3-series -> NVIDIA Tesla T4 -> Turing
# https://developer.nvidia.com/blog/nvidia-releases-open-source-gpu-kernel-modules/
sudo apt-get update
sudo apt-get install -y \
    linux-modules-nvidia-$NVIDIA_DRIVER_BRANCH-server-open-$(uname -r) \
    nvidia-headless-no-dkms-$NVIDIA_DRIVER_BRANCH-server-open \
    nvidia-utils-$NVIDIA_DRIVER_BRANCH-server
sudo modprobe nvidia

# Install Miniconda3
download "${BUILD_DIR}/${MINICONDA3_INSTALLER_FILE_NAME}" "${MINICONDA3_DOWNLOAD_URL}"
bash "${BUILD_DIR}/${MINICONDA3_INSTALLER_FILE_NAME}" -b -u -p "${BUILD_DIR}/miniconda3"

# Setup Miniconda3
$BUILD_DIR/miniconda3/bin/conda init bash
eval "$($BUILD_DIR/miniconda3/bin/conda shell.bash hook)"
conda create -y -n build python="${MINICONDA3_PYTHON_MAJOR_VERSION}.${MINICONDA3_PYTHON_MINOR_VERSION}"
conda activate build

# Install Pytorch
sudo apt install -y gcc-14  # must install c compiler for pytorch model compiling
pip install torch torchvision

# Install diffusers
pip install git+https://github.com/huggingface/diffusers transformers accelerate datasets bitsandbytes wandb

# Install SD3 deps
pip install sentencepiece protobuf
