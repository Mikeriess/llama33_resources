import os
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    
    print(f"Loading base model: {hyperparams['model']['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        hyperparams["model"]["base_model"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter: {lora_model_id}")
    adapter_model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("Merging LoRA weights into base model...")
    merged_model = adapter_model.merge_and_unload()
    
    # Get model name for saving
    model_name = lora_model_id.split("/")[-1]
    merged_model_path = f"merges/{model_name}-merged"
    
    print(f"Saving merged model to: {merged_model_path}")
    merged_model.save_pretrained(
        merged_model_path,
        safe_serialization=True
    )
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_model_id)
    tokenizer.save_pretrained(merged_model_path)
    
    # Upload to Hub
    hub_model_id = f"MykMaks/{model_name}-merged"
    print(f"Uploading merged model to Hub: {hub_model_id}")
    
    merged_model.push_to_hub(
        hub_model_id,
        safe_serialization=True,
        private=False
    )
    tokenizer.push_to_hub(hub_model_id)
    
    print("Done! Model merged and uploaded successfully.")
    return hub_model_id

def main():
    # Load hyperparameters
    hyperparams = load_hyperparams()
    
    # LoRA model to merge
    lora_model_id = "MykMaks/llama-3.2-11B-MM-20-MykMaks_da-wit"
    
    # Merge and upload
    merged_model_id = merge_and_upload_model(lora_model_id, hyperparams)
    print(f"Model available at: https://huggingface.co/{merged_model_id}")

if __name__ == "__main__":
    main()