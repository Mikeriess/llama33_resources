import argparse
from datetime import datetime
import wandb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Finetune Vision Language Model')
    parser.add_argument('--data', type=str, default="unsloth/Radiology_mini",
                      help='Dataset name to use for finetuning (default: unsloth/Radiology_mini)')
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='Local rank for distributed training')
    return parser.parse_args()

instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["caption"]} ]
        },
    ]
    return { "messages" : conversation }

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    args = parse_args()
    
    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:
        setup_ddp(local_rank, world_size)
        torch.cuda.set_device(local_rank)

    # Only initialize wandb on the main process
    if local_rank <= 0:
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        wandb.init(
            project="hack_oslo",
            name=current_time,
            anonymous="allow",
            config={
                # Dataset config
                "dataset": args.data,
                "instruction": instruction,
                
                # Model config
                "model_name": "unsloth/Llama-3.2-11B-Vision-Instruct",
                "load_in_4bit": True,  #TODO: False
                "use_gradient_checkpointing": "unsloth",
                
                # LoRA config
                "finetune_vision_layers": False,
                "finetune_language_layers": True,
                "finetune_attention_modules": True,
                "finetune_mlp_modules": True,
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_bias": "none",
                "random_state": 1337,
                "use_rslora": False,
                
                # Training config
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "warmup_steps": 5,
                "max_steps": 30,
                "weight_decay": 0.01,
                "lr_scheduler": "linear",
                "fp16": not is_bf16_supported(),
                "bf16": is_bf16_supported(),
                "max_seq_length": 2048,
                "num_workers": 8,
                "num_gpus": world_size,
            }
        )

    from unsloth import FastVisionModel, is_bf16_supported
    from datasets import load_dataset
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    # Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map="auto",  # Automatically handle multi-GPU
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=wandb.config.finetune_vision_layers if local_rank <= 0 else False,
        finetune_language_layers=wandb.config.finetune_language_layers if local_rank <= 0 else True,
        finetune_attention_modules=wandb.config.finetune_attention_modules if local_rank <= 0 else True,
        finetune_mlp_modules=wandb.config.finetune_mlp_modules if local_rank <= 0 else True,
        r=wandb.config.lora_r if local_rank <= 0 else 16,
        lora_alpha=wandb.config.lora_alpha if local_rank <= 0 else 16,
        lora_dropout=wandb.config.lora_dropout if local_rank <= 0 else 0.05,
        bias=wandb.config.lora_bias if local_rank <= 0 else "none",
        random_state=wandb.config.random_state if local_rank <= 0 else 1337,
        use_rslora=wandb.config.use_rslora if local_rank <= 0 else False,
        loftq_config=None,
    )

    # Load dataset
    dataset = load_dataset(args.data, split="train")
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    FastVisionModel.for_training(model)

    # Configure trainer for distributed training
    training_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1337,
        output_dir="outputs",
        report_to="wandb" if local_rank <= 0 else "none",
        save_strategy="steps",
        save_steps=30,
        save_total_limit=1,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=8,
        max_seq_length=2048,
        
        # DDP specific settings
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=training_args,
    )

    # Print GPU stats only on main process
    if local_rank <= 0:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        print(f"Training on {world_size} GPUs")

    trainer_stats = trainer.train()
    
    if local_rank <= 0:
        wandb.finish()

    if local_rank != -1:
        cleanup_ddp()

if __name__ == "__main__":
    main() 