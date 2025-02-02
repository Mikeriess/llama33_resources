import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import json
from huggingface_hub import HfApi
import torch

def load_hyperparams(file_path="hyperparams.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

def merge_and_upload_model(lora_model_id, hyperparams):
    # Create merges directory if it doesn't exist
    os.makedirs("merges", exist_ok=True)
    
    # Setup quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print(f"Loading base model: {hyperparams['model']['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        hyperparams["model"]["base_model"],
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    
    print(f"Loading LoRA adapter: {lora_model_id}")
    adapter_model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        is_trainable=False  # Prevent quantization issues
    )
    
    print("Merging LoRA weights into base model...")
    merged_model = adapter_model.merge_and_unload()
    
    # Get model name for saving
    model_name = lora_model_id.split("/")[-1]
    merged_model_path = f"merges/{model_name}-merged"
    
    print(f"Saving unquantized merged model to: {merged_model_path}")
    # Save unquantized version
    merged_model.save_pretrained(
        merged_model_path,
        safe_serialization=True,
        max_shard_size="5GB"  # Add sharding for large models
    )
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_model_id)
    tokenizer.save_pretrained(merged_model_path)
    
    # Upload to Hub
    hub_model_id = f"MykMaks/{model_name}-merged"
    print(f"Uploading merged model to Hub: {hub_model_id}")
    
    # Upload unquantized version with proper sharding
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
    
    # Get list of all LoRA models to merge
    api = HfApi()
    models = api.list_models(author="MykMaks", search="llama-3.2-11B-MM")
    lora_models = [model.modelId for model in models]
    
    print(f"Found {len(lora_models)} LoRA models to merge")
    
    for lora_model_id in lora_models:
        try:
            print(f"\nProcessing model: {lora_model_id}")
            merged_model_id = merge_and_upload_model(lora_model_id, hyperparams)
            print(f"Successfully merged and uploaded: https://huggingface.co/{merged_model_id}")
        except Exception as e:
            print(f"Error processing {lora_model_id}: {str(e)}")
            continue

if __name__ == "__main__":
    main()