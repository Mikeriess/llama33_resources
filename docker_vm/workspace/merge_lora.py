import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import json
import torch
from huggingface_hub import HfApi

def load_hyperparams(file_path="hyperparams.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

def merge_and_upload_model(lora_model_id, hyperparams):
    """Merge LoRA adapter with base model following best practices"""
    
    # Create merges directory if it doesn't exist
    os.makedirs("merges", exist_ok=True)
    
    # 1. First load base model without quantization
    print(f"Loading base model: {hyperparams['model']['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        hyperparams["model"]["base_model"],
        device_map="auto",
        torch_dtype=torch.float16,  # Use same dtype as training
    )
    
    # 2. Load adapter model
    print(f"Loading LoRA adapter: {lora_model_id}")
    adapter_model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_id,
        device_map="auto",
        torch_dtype=torch.float16,  # Use same dtype as training
        is_trainable=False  # Prevent quantization issues
    )
    
    # 3. Merge adapter with base model
    print("Merging LoRA weights into base model...")
    merged_model = adapter_model.merge_and_unload()
    
    # Get model name for saving
    model_name = lora_model_id.split("/")[-1]
    merged_model_path = f"merges/{model_name}-merged"
    
    # 4. Save unquantized merged model
    print(f"Saving unquantized merged model to: {merged_model_path}")
    merged_model.save_pretrained(
        merged_model_path,
        safe_serialization=True,
        max_shard_size="5GB"  # Shard large models
    )
    
    # 5. Save tokenizer with all necessary files
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model"]["base_model"])
    tokenizer.save_pretrained(merged_model_path)
    
    # Ensure all tokenizer files are present
    required_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json"
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(merged_model_path, file)):
            source_path = os.path.join(hyperparams["model"]["base_model"], file)
            if os.path.exists(source_path):
                import shutil
                shutil.copy2(source_path, merged_model_path)
    
    # 6. Upload to Hub
    hub_model_id = f"MykMaks/{model_name}-merged"
    print(f"Uploading merged model to Hub: {hub_model_id}")
    
    merged_model.push_to_hub(
        hub_model_id,
        safe_serialization=True,
        max_shard_size="5GB",
        private=False
    )
    tokenizer.push_to_hub(hub_model_id)
    
    print("Done! Model merged and uploaded successfully.")
    return hub_model_id

def main():
    # Load hyperparameters
    hyperparams = load_hyperparams()
    
    # Specific LoRA model to merge
    lora_model_id = "MykMaks/llama-3.2-11B-MM-20-MykMaks_da-wit"
    
    try:
        print(f"\nProcessing model: {lora_model_id}")
        merged_model_id = merge_and_upload_model(lora_model_id, hyperparams)
        print(f"Successfully merged and uploaded: https://huggingface.co/{merged_model_id}")
    except Exception as e:
        print(f"Error processing {lora_model_id}: {str(e)}")
        raise

if __name__ == "__main__":
    main()