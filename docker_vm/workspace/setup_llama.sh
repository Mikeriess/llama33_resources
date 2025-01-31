#!/bin/bash

# Install requirements
echo "Installing Python packages..."
pip3 install -r requirements.txt

# Login to wandb
echo "Logging into Weights & Biases..."
wandb login

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.4 support..."
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
echo "Verifying CUDA installation..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"

# Create models directory
echo "Creating models directory..."
mkdir -p models

# Create data directory
echo "Creating data directory..."
mkdir -p data

echo "Setup complete! You can now run the training script with:"
echo "python3 finetune_llama.py" 