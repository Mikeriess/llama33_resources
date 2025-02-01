import argparse
from datetime import datetime
import wandb
import os
import torch
import json

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

def main():
    args = parse_args()
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Load hyperparameters
    hyperparams = load_hyperparams(args.hyperparams)
    
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    from unsloth import FastVisionModel, is_bf16_supported
    
    # Check for local checkpoint
    local_checkpoint = "llama32_checkpoint"
    base_model = hyperparams["model"]["base_model"]
    
    # Initialize model
    if os.path.exists(local_checkpoint):
        print(f"Loading from local checkpoint: {local_checkpoint}")
        model, tokenizer = FastVisionModel.from_pretrained(
            local_checkpoint,
            load_in_4bit=hyperparams["model"]["load_in_4bit"],
            use_gradient_checkpointing=hyperparams["model"]["use_gradient_checkpointing"],
            device_map="auto",
        )
        model_source = local_checkpoint
    else:
        print(f"Loading base model: {base_model}")
        model, tokenizer = FastVisionModel.from_pretrained(
            base_model,
            load_in_4bit=hyperparams["model"]["load_in_4bit"],
            use_gradient_checkpointing=hyperparams["model"]["use_gradient_checkpointing"],
            device_map="auto",
        )
        # Only add LoRA adapters when starting from base model
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=hyperparams["lora"]["finetune_vision_layers"],
            finetune_language_layers=hyperparams["lora"]["finetune_language_layers"],
            finetune_attention_modules=hyperparams["lora"]["finetune_attention_modules"],
            finetune_mlp_modules=hyperparams["lora"]["finetune_mlp_modules"],
            r=hyperparams["lora"]["r"],
            lora_alpha=hyperparams["lora"]["alpha"],
            lora_dropout=hyperparams["lora"]["dropout"],
            bias=hyperparams["lora"]["bias"],
            random_state=hyperparams["lora"]["random_state"],
            use_rslora=hyperparams["lora"]["use_rslora"],
            loftq_config=None,
        )
        model_source = base_model

    print(f"Loading model from: {model_source}")
    
    # Calculate batch size and gradient accumulation based on number of GPUs
    batch_size = hyperparams["training"]["base_batch_size"] * num_gpus
    grad_accum = max(1, hyperparams["training"]["base_gradient_accumulation_steps"] // num_gpus)
    
    wandb.init(
        project="hack_oslo",
        name=current_time,
        anonymous="allow",
        config={
            # Dataset config
            "dataset": args.data,
            "instruction": args.instruction,
            "model_source": model_source,
            "base_model": base_model,
            
            # Model config
            **hyperparams["model"],
            
            # LoRA config
            **hyperparams["lora"],
            
            # Training config
            **hyperparams["training"],
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "fp16": not is_bf16_supported(),
            "bf16": is_bf16_supported(),
            "num_gpus": num_gpus,
        }
    )

    from datasets import load_dataset
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers.trainer_utils import get_last_checkpoint

    dataset = load_dataset(wandb.config.dataset, split="train")
    converted_dataset = [convert_to_conversation(sample, wandb.config.instruction, args.text_field) for sample in dataset]

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

    trainer_stats = trainer.train()
    
    # Save model locally
    print("Saving model locally...")
    model.save_pretrained(local_checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(local_checkpoint)
    
    # Upload to HuggingFace Hub
    #if "HF_TOKEN" in os.environ:
    print("Uploading model to HuggingFace Hub...")
    repo_id = f"MykMaks/llama32_{args.experiment_number}_{args.data.replace('/', '_')}"
    print(f"Repo ID: {repo_id}")
    model.push_to_hub(repo_id, safe_serialization=True)
    tokenizer.push_to_hub(repo_id)
    #else:
    #    print("Warning: HF_TOKEN not found in environment, skipping upload")
    
    wandb.finish()

if __name__ == "__main__":
    main() 