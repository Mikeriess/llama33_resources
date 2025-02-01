import argparse
from datetime import datetime
import wandb
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Finetune Vision Language Model')
    parser.add_argument('--data', type=str, default="unsloth/Radiology_mini",
                      help='Dataset name to use for finetuning (default: unsloth/Radiology_mini)')
    parser.add_argument('--instruction', type=str, 
                      default="You are an expert radiographer. Describe accurately what you see in this image.",
                      help='Instruction prompt for the model')
    parser.add_argument('--experiment_number', type=int, required=True,
                      help='Experiment number for model naming')
    return parser.parse_args()

def convert_to_conversation(sample, instruction):
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

def main():
    args = parse_args()
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    from unsloth import FastVisionModel, is_bf16_supported
    
    # Check for local checkpoint
    local_checkpoint = "llama32_checkpoint"
    model_id = local_checkpoint if os.path.exists(local_checkpoint) else "unsloth/Llama-3.2-11B-Vision-Instruct"
    print(f"Loading model from: {model_id}")
    
    wandb.init(
        project="hack_oslo",
        name=current_time,
        anonymous="allow",
        config={
            # Dataset config
            "dataset": args.data,
            "instruction": args.instruction,
            "model_source": model_id,
            
            # Model config
            "model_name": "unsloth/Llama-3.2-11B-Vision-Instruct",
            "load_in_4bit": True,
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
            "batch_size": 2 * num_gpus,  # Scale batch size with number of GPUs
            "gradient_accumulation_steps": max(1, 4 // num_gpus),  # Adjust accumulation based on GPUs
            "learning_rate": 2e-4,
            "warmup_steps": 5,
            "max_steps": 30,
            "weight_decay": 0.01,
            "lr_scheduler": "linear",
            "fp16": not is_bf16_supported(),
            "bf16": is_bf16_supported(),
            "max_seq_length": 2048,
            "num_workers": 8,
            "num_gpus": num_gpus,
        }
    )

    from datasets import load_dataset
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers.trainer_utils import get_last_checkpoint

    # Load base model (either local checkpoint or original model)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=wandb.config.load_in_4bit,
        use_gradient_checkpointing=wandb.config.use_gradient_checkpointing,
        device_map="auto",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=wandb.config.finetune_vision_layers,
        finetune_language_layers=wandb.config.finetune_language_layers,
        finetune_attention_modules=wandb.config.finetune_attention_modules,
        finetune_mlp_modules=wandb.config.finetune_mlp_modules,
        r=wandb.config.lora_r,
        lora_alpha=wandb.config.lora_alpha,
        lora_dropout=wandb.config.lora_dropout,
        bias=wandb.config.lora_bias,
        random_state=wandb.config.random_state,
        use_rslora=wandb.config.use_rslora,
        loftq_config=None,
    )

    dataset = load_dataset(wandb.config.dataset, split="train")
    converted_dataset = [convert_to_conversation(sample, wandb.config.instruction) for sample in dataset]

    FastVisionModel.for_training(model)

    training_args = SFTConfig(
        per_device_train_batch_size=wandb.config.batch_size // num_gpus,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        warmup_steps=wandb.config.warmup_steps,
        max_steps=wandb.config.max_steps,
        learning_rate=wandb.config.learning_rate,
        fp16=wandb.config.fp16,
        bf16=wandb.config.bf16,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=wandb.config.weight_decay,
        lr_scheduler_type=wandb.config.lr_scheduler,
        seed=wandb.config.random_state,
        output_dir="outputs",
        report_to="wandb",
        save_strategy="steps",
        save_steps=wandb.config.max_steps,
        save_total_limit=1,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=wandb.config.num_workers,
        max_seq_length=wandb.config.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=training_args,
    )

    # Print GPU stats
    # for i in range(num_gpus):
    #     gpu_stats = torch.cuda.get_device_properties(i)
    #     max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    #     print(f"GPU {i} = {gpu_stats.name}. Max memory = {max_memory} GB.")
    # print(f"Training on {num_gpus} GPUs")

    trainer_stats = trainer.train()
    
    # Save model locally - merge adapter weights and save full model
    print("Saving model locally...")
    merged_model = model.merge_and_unload()  # This merges the LoRA weights into the base model
    merged_model.save_pretrained(local_checkpoint)
    tokenizer.save_pretrained(local_checkpoint)
    
    # Upload to HuggingFace Hub - also upload the merged model
    if "HF_TOKEN" in os.environ:
        print("Uploading model to HuggingFace Hub...")
        repo_id = f"llama32_{args.experiment_number}_{args.data.replace('/', '_')}"
        merged_model.push_to_hub(repo_id, use_auth_token=os.environ["HF_TOKEN"])
        tokenizer.push_to_hub(repo_id, use_auth_token=os.environ["HF_TOKEN"])
    else:
        print("Warning: HF_TOKEN not found in environment, skipping upload")
    
    # Clean up to free memory
    del merged_model
    torch.cuda.empty_cache()
    
    wandb.finish()

if __name__ == "__main__":
    main() 