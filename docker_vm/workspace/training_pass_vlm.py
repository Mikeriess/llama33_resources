import argparse
from datetime import datetime
import wandb
import os
import torch
import json
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import BitsAndBytesConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Finetune Vision Language Model')
    parser.add_argument('--data', type=str, required=True,
                      help='Dataset name to use for finetuning')
    parser.add_argument('--instruction', type=str, required=True,
                      help='Instruction prompt for the model')
    parser.add_argument('--text_field', type=str, required=True,
                      help='Field name containing the text in the dataset')
    parser.add_argument('--experiment_number', type=int, required=True,
                      help='Experiment number for model naming')
    parser.add_argument('--hyperparams', type=str, default="hyperparams.json",
                      help='Path to hyperparameters configuration file')
    # Add distributed training arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='Local rank for distributed training')
    return parser.parse_args()

def load_hyperparams(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def convert_to_conversation(sample, instruction, text_field):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample[text_field]} ]
        },
    ]
    return { "messages" : conversation }

def setup_distributed():
    # Initialize the distributed environment
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = torch.cuda.device_count()
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    return dist.get_rank(), world_size

def get_model_config(hyperparams):
    # Create quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Model loading config
    model_config = {
        "quantization_config": bnb_config,
        "device_map": "auto",  # Let transformers handle device mapping
        "torch_dtype": torch.float16
    }
    
    return model_config

def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    is_main_process = rank == 0
    
    # Load hyperparameters
    hyperparams = load_hyperparams(args.hyperparams)
    model_config = get_model_config(hyperparams)
    
    if is_main_process:
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        wandb.init(
            project="hack_oslo",
            name=current_time,
            config={
                "dataset": args.data,
                "instruction": args.instruction,
                **hyperparams,
                "world_size": world_size,
                "quantization": model_config["quantization_config"].to_dict()
            }
        )
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Initialize model
    local_checkpoint = "llama32_checkpoint"
    base_model = hyperparams["model"]["base_model"]
    
    if os.path.exists(local_checkpoint):
        print(f"[{rank}] Loading from checkpoint: {local_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            local_checkpoint,
            **model_config
        )
        tokenizer = AutoTokenizer.from_pretrained(local_checkpoint)
    else:
        print(f"[{rank}] Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_config
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Add LoRA config
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=hyperparams["lora"]["r"],
            lora_alpha=hyperparams["lora"]["alpha"],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=hyperparams["lora"]["dropout"],
            bias=hyperparams["lora"]["bias"],
        )
        model = get_peft_model(model, lora_config)
    
    # Wrap model in DDP after quantization
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Load and prepare dataset
    from datasets import load_dataset
    dataset = load_dataset(args.data, split="train")
    converted_dataset = [convert_to_conversation(sample, args.instruction, args.text_field) 
                        for sample in dataset]
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        converted_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Training configuration
    from transformers import Trainer, TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=hyperparams["training"]["base_batch_size"],
        gradient_accumulation_steps=hyperparams["training"]["base_gradient_accumulation_steps"],
        learning_rate=hyperparams["training"]["learning_rate"],
        max_steps=hyperparams["training"]["max_steps"],
        bf16=True,  # Use bfloat16 for training
        logging_steps=1,
        save_strategy="steps",
        save_steps=hyperparams["training"]["max_steps"],
        ddp_find_unused_parameters=False,
        report_to="wandb" if is_main_process else None,
        local_rank=args.local_rank,
        # Add gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
        # Add proper fp16 settings
        fp16=False,  # Disable fp16 since we're using bf16
        tf32=True,  # Enable tensor float 32 for better performance on Ampere GPUs
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=converted_dataset,
        data_collator=DefaultDataCollator(),
        train_sampler=train_sampler,  # Add distributed sampler
    )
    
    trainer.train()
    
    if is_main_process:
        # Save model locally
        model.save_pretrained(local_checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(local_checkpoint)
        
        # Upload to HuggingFace Hub
        repo_id = f"MykMaks/llama-3.2-11B-V-I_{args.experiment_number}_{args.data.replace('/', '_')}"
        model.push_to_hub(repo_id, safe_serialization=True)
        tokenizer.push_to_hub(repo_id)
        
        wandb.finish()
    
    # Clean up distributed training
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 