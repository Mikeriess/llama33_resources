#!/bin/bash

# Create and activate a virtual environment
echo "Creating virtual environment..."
python3 -m venv llama_env
source llama_env/bin/activate

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.4 support..."
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
echo "Verifying CUDA installation..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"

# Create models directory
echo "Creating models directory..."
mkdir -p models

echo "Setup complete! You can now use the virtual environment with:"
echo "source llama_env/bin/activate" 