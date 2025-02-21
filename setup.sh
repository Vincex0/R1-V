#!/bin/bash

# Ensure specific CUDA version
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create and activate a new virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip and install pip-tools
pip install --upgrade pip==24.0
pip install pip-tools==7.3.0

# Install PyTorch and CUDA packages first
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention with specific CUDA architecture
export TORCH_CUDA_ARCH_LIST="8.0"
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# Install remaining CUDA-specific requirements
pip install -r cuda_requirements.txt

# Install other requirements
pip install -r requirements_full.txt

# Install the local package
cd src/r1-v 
pip install -e ".[dev]"

# Verify installations
echo "=== System CUDA Version ==="
nvidia-smi | grep 'CUDA Version'

echo -e "\n=== PyTorch and CUDA Info ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}\nCUDA version: {torch.version.cuda}\nCUDA available: {torch.cuda.is_available()}\nGPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

echo -e "\n=== Flash Attention Info ==="
python3 -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"

echo -e "\n=== CUDA Environment Variables ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# vLLM support 
pip install vllm==0.7.2

# fix transformers version
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

pip uninstall -y flash-attn && TORCH_CUDA_ARCH_LIST="8.0" pip install flash-attn --no-build-isolation --no-cache-dir