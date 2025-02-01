import argparse
from datetime import datetime
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune Vision Language Model')
    parser.add_argument('--data', type=str, default="unsloth/Radiology_mini",
                      help='Dataset name to use for finetuning (default: unsloth/Radiology_mini)')
    parser.add_argument('--project', type=str, default=None,
                      help='Project name for Weights & Biases (default: metahack_<datetime>)')
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

def main():
    from unsloth import FastVisionModel
    import torch
    args = parse_args()

    # Setup W&B project name
    if args.project is None:
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        project_name = f"metahack_{current_time}"
    else:
        project_name = f"metahack_{args.project}"

    # Initialize W&B
    wandb.init(
        project=project_name,
        config={
            "dataset": args.data,
            "model": "unsloth/Llama-3.2-11B-Vision-Instruct",
            "load_in_4bit": True,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 5,
            "max_steps": 30,
        }
    )

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    from datasets import load_dataset
    dataset = load_dataset(args.data, split = "train")

    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    FastVisionModel.for_inference(model) # Enable for inference!

    image = dataset[0]["image"]
    instruction = "You are an expert radiographer. Describe accurately what you see in this image."

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

    from unsloth import is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model) # Enable for training!

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = converted_dataset,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 30,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb",  # Changed from "none" to "wandb"

            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
    )

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
