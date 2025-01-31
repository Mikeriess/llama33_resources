import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import argparse

class SyntheticQAGenerator:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B", device="cuda"):
        """Initialize the QA generator with specified model."""
        print(f"Loading model {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device = device

    def generate_qa_pair(self, context, max_length=512):
        """Generate a question-answer pair based on the given context."""
        prompt = f"""Based on the following text, generate a relevant question and its detailed answer in Norwegian. 
        Format the output as JSON with 'question' and 'answer' fields.
        
        Text: {context}
        
        Generate QA pair:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            # Extract the JSON part from the response
            json_str = response.split("Generate QA pair:")[-1].strip()
            qa_pair = json.loads(json_str)
            return {
                "prompt": qa_pair["question"],
                "response": qa_pair["answer"]
            }
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

def load_sources(file_path):
    """Load source documents from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_qa_pairs(qa_pairs, output_file):
    """Save generated QA pairs to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"train": qa_pairs}, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic QA pairs from source documents')
    parser.add_argument('--model', default="meta-llama/Meta-Llama-3.1-8B", help='Model ID to use for generation')
    parser.add_argument('--num_pairs', type=int, default=3, help='Number of QA pairs to generate per source')
    parser.add_argument('--output', default='data/synthetic_qa_pairs.json', help='Output file path')
    args = parser.parse_args()

    # Ensure input/output directories exist
    os.makedirs('data/input', exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize generator
    generator = SyntheticQAGenerator(model_id=args.model)

    # Load source documents
    try:
        sources = load_sources('data/input/sources.json')
    except FileNotFoundError:
        print("Error: sources.json not found in data/input/. Creating example file...")
        example_sources = {
            "sources": [
                {
                    "title": "Example Norwegian History",
                    "content": "Norge ble forent til ett rike under Harald Hårfagre på 800-tallet. Dette markerte begynnelsen på vikingtiden i norsk historie."
                }
            ]
        }
        os.makedirs('data/input', exist_ok=True)
        with open('data/input/sources.json', 'w', encoding='utf-8') as f:
            json.dump(example_sources, f, ensure_ascii=False, indent=4)
        sources = example_sources

    # Generate QA pairs
    qa_pairs = []
    for source in tqdm(sources["sources"], desc="Generating QA pairs"):
        for _ in range(args.num_pairs):
            qa_pair = generator.generate_qa_pair(source["content"])
            if qa_pair:
                qa_pairs.append(qa_pair)

    # Save generated pairs
    save_qa_pairs(qa_pairs, args.output)
    print(f"Generated {len(qa_pairs)} QA pairs and saved to {args.output}")

if __name__ == "__main__":
    main() 