import argparse
from unsloth import FastVisionModel
import torch
from transformers import TextStreamer

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with Vision Language Model')
    parser.add_argument('--model_path', type=str, default="unsloth/Llama-3.2-11B-Vision-Instruct",
                      help='Path to model or model identifier from huggingface.co/models')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to image file for inference')
    parser.add_argument('--instruction', type=str, 
                      default="You are an expert radiographer. Describe accurately what you see in this image.",
                      help='Instruction prompt for the model')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # Enable inference mode
    FastVisionModel.for_inference(model)

    # Load and process image
    print(f"Loading image from {args.image_path}...")
    from PIL import Image
    image = Image.open(args.image_path)

    # Prepare input
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": args.instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Run inference
    print("\nGenerating response...")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )
    print("\n")

if __name__ == "__main__":
    main() 