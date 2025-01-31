# Norwegian Language Model Fine-tuning Pipeline

This repository contains a complete pipeline for generating, evaluating, and fine-tuning Norwegian language models using LLaMA 3.1. The pipeline consists of three main components:

## 1. Synthetic Data Generation
**Tool:** synthetic_generator.py  
**Purpose:** Generates Norwegian question-answer pairs from source documents  
**Input:** Norwegian text documents in data/input/sources.json  
**Output:** Generated QA pairs in data/output/synthetic_qa_pairs.json  
**Usage:** `python3 synthetic_generator.py [--model MODEL_ID] [--num_pairs N] [--output OUTPUT_FILE]`

## 2. Quality Evaluation
**Tool:** reward_model.py  
**Purpose:** Evaluates the quality of generated QA pairs using NVIDIA's reward model  
**Input:** QA pairs from data/output/synthetic_qa_pairs.json  
**Output:** Evaluated pairs with scores in data/output/evaluated_qa_pairs.json  
**Usage:** `python3 reward_model.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--model MODEL_ID]`

## 3. Model Fine-tuning
**Tool:** finetune_llama.py  
**Purpose:** Fine-tunes LLaMA model on high-quality Norwegian QA pairs  
**Input:** Training data in data/sample_dataset.json  
**Output:** Fine-tuned model in ./results directory  
**Usage:** `python3 finetune_llama.py`

## Setup and Requirements
In JupyterLab terminal, run:   ./setup_llama.sh

# REQUIREMENTS:
- Hugging Face account with model access
- Weights & Biases account for logging

# DIRECTORY STRUCTURE:
```
docker_vm/
├── workspace/           # Main working directory
│   ├── data/           # Data directory
│   │   ├── input/      # Source documents
│   │   └── output/     # Generated and evaluated QA pairs
│   ├── results/        # Fine-tuned models
│   └── models/         # Downloaded model weights
└── Dockerfile          # Container configuration
```
# WORKFLOW:
1. Place Norwegian source documents in data/input/sources.json
2. Generate QA pairs: python3 synthetic_generator.py
3. Evaluate quality: python3 reward_model.py
4. Fine-tune model: python3 finetune_llama.py
5. Monitor training on Weights & Biases dashboard

# EXAMPLE MODELS:
- Generation: meta-llama/Meta-Llama-3.1-8B
- Evaluation: nvidia/Llama-3.1-Nemotron-70B-Reward-HF
- Fine-tuning base: meta-llama/Meta-Llama-3.1-8B

NOTES:
- All scripts include detailed help: python3 script_name.py --help
- See individual .txt files for detailed documentation of each component
- Memory requirements vary by model size and batch settings
- Default settings optimized for RTX 4090 (24GB VRAM)
- Uses LoRA and 4-bit quantization for efficient training 
