import os
from transformers import AutoTokenizer
from unsloth import FastVisionModel, is_bf16_supported
import json
import torch
from huggingface_hub import HfApi

def load_hyperparams(file_path="hyperparams.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def merge_and_upload_model(lora_model_id, hyperparams):
    """Merge LoRA adapter with base model following unsloth best practices"""
    
    # Create merges directory if it doesn't exist
    os.makedirs("merges", exist_ok=True)
    
    # 1. First load base model - using same initialization as training_pass_vlm.py
    print(f"Loading base model: {hyperparams['model']['base_model']}")
    base_model, tokenizer = FastVisionModel.from_pretrained(
        hyperparams["model"]["base_model"],
        load_in_4bit=hyperparams["model"]["load_in_4bit"],
        use_gradient_checkpointing=hyperparams["model"]["use_gradient_checkpointing"],
        device_map="auto",
    )
    
    # 2. Load adapter model with existing LoRA weights
    print(f"Loading LoRA adapter: {lora_model_id}")
    adapter_model, _ = FastVisionModel.from_pretrained(
        lora_model_id,  # This already has LoRA weights
        load_in_4bit=hyperparams["model"]["load_in_4bit"],
        use_gradient_checkpointing=hyperparams["model"]["use_gradient_checkpointing"],
        device_map="auto",
    )
    
    # 3. Merge adapter with base model
    print("Merging LoRA weights into base model...")
    merged_model = adapter_model.merge_and_unload()
    
    # Get model name for saving
    model_name = lora_model_id.split("/")[-1]
    merged_model_path = f"merges/{model_name}-merged"
    
    # 4. Save merged model
    print(f"Saving merged model to: {merged_model_path}")
    merged_model.save_pretrained(
        merged_model_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # 5. Save tokenizer with all necessary files
    print("Saving tokenizer...")
    tokenizer.save_pretrained(merged_model_path)
    
    # Ensure all tokenizer files are present
    required_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "config.json"
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

    # save quantized verisons of the model
    print(f"Uploading merged model to Hub in 4bit: {hub_model_id}")
    #merged_model.push_to_hub_gguf(hub_model_id+"-q4_k_m", tokenizer, quantization_method = "q4_k_m")
    #merged_model.push_to_hub_gguf(hub_model_id+"-q8_0", tokenizer, quantization_method = "q8_0")
    merged_model.save_pretrained_merged(hub_model_id+"-4bit", tokenizer, save_method = "merged_4bit",)
    
    print("Done! Model merged and uploaded successfully.")
    return hub_model_id

def main():
    # Load hyperparameters - same as training
    hyperparams = load_hyperparams()
    
    # Specific LoRA model to merge
    #lora_model_id = "MykMaks/llama-3.2-11B-MM-20-MykMaks_da-wit"
    lora_model_id = "MykMaks/llama-3.2-11B-V-I_39_MykMaks_NorwegianDataset-compressed-pt2"
    try:
        print(f"\nProcessing model: {lora_model_id}")
        merged_model_id = merge_and_upload_model(lora_model_id, hyperparams)
        print(f"Successfully merged and uploaded: https://huggingface.co/{merged_model_id}")
    except Exception as e:
        print(f"Error processing {lora_model_id}: {str(e)}")
        raise

if __name__ == "__main__":
    main()