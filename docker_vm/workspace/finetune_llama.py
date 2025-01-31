import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import json
from huggingface_hub import login
import wandb

def load_dataset_from_json(file_path):
    """Load and format the dataset from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data['train']:
        formatted_data.append({
            "text": f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>\n"
        })
    
    return Dataset.from_dict({"text": [d["text"] for d in formatted_data]})

def setup_model_and_tokenizer(model_id):
    """Setup the model and tokenizer with 4-bit quantization."""
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    return model, tokenizer

def main():
    # Initialize wandb
    wandb.init(
        project="llama-norwegian-ft",
        config={
            "model": "meta-llama/Meta-Llama-3.1-8B",
            "learning_rate": 2e-4,
            "batch_size": 4,
            "epochs": 3,
            "lora_r": 8,
            "lora_alpha": 16,
        }
    )

    # Login to Hugging Face (you'll need to have your token ready)
    login()

    # Configuration
    MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
    OUTPUT_DIR = "./results"
    DATA_DIR = "./data"  # Add data directory path
    
    # Load dataset
    dataset = load_dataset_from_json(f"{DATA_DIR}/sample_dataset.json")  # Update dataset path
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_ID)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # Target attention layers only
    )
    
    # Training arguments optimized for RTX 4090
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        max_steps=100,
        fp16=True,
        push_to_hub=False,  # Set to True if you want to push to HF Hub
        # Add wandb reporting
        report_to="wandb",
        # Add run name for wandb
        run_name=f"llama-norwegian-ft-{wandb.run.id}"
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=512,
        packing=False,
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 