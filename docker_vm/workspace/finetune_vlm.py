import argparse
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune Vision Language Model')
    parser.add_argument('--data', type=str, default="unsloth/Radiology_mini",
                      help='Dataset name to use for finetuning (default: unsloth/Radiology_mini)')
    return parser.parse_args()

def convert_to_conversation(sample):
    instruction = "You are an expert radiographer. Describe accurately what you see in this image."
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

def test_inference(model, tokenizer, dataset):
    """Run a test inference like in the notebook"""
    FastVisionModel.for_inference(model)
    
    image = dataset[0]["image"]
    instruction = "You are an expert radiographer. Describe accurately what you see in this image."
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                      use_cache=True, temperature=1.5, min_p=0.1)

def main():
    args = parse_args()
    
    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

        "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    dataset = load_dataset(args.data, split = "train")
    
    # Optional: Display first image and caption like in notebook
    print("\nDataset sample:")
    print(f"Caption for first image: {dataset[0]['caption']}\n")
    
    # Convert dataset
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    print("First converted sample:")
    print(converted_dataset[0])
    print()
    
    # Run test inference like in notebook
    test_inference(model, tokenizer, dataset)

    # Print GPU stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Setup trainer
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
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
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        ),
    )

    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()

    # Print final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory/max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"\nTraining completed!")
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

if __name__ == "__main__":
    main() 