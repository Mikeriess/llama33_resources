import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
import argparse
from pathlib import Path

class QAEvaluator:
    def __init__(self, model_id="nvidia/Llama-3.1-Nemotron-70B-Reward-HF", device="cuda"):
        """Initialize the reward model for QA evaluation."""
        print(f"Loading reward model {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = device

    def evaluate_qa_pair(self, prompt, response):
        """Evaluate a single QA pair using the reward model."""
        messages = [
            {'role': "user", "content": prompt},
            {'role': "assistant", "content": response}
        ]
        
        tokenized_message = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        )
        
        with torch.no_grad():
            response_token_ids = self.model.generate(
                tokenized_message['input_ids'].cuda(),
                attention_mask=tokenized_message['attention_mask'].cuda(),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            reward = response_token_ids['scores'][0][0][0].item()
        
        return reward

def load_qa_pairs(file_path):
    """Load QA pairs from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['train']

def save_evaluated_pairs(qa_pairs, scores, output_file):
    """Save QA pairs with their reward scores."""
    output_data = {
        "train": [
            {
                "prompt": pair["prompt"],
                "response": pair["response"],
                "reward_score": score
            }
            for pair, score in zip(qa_pairs, scores)
        ]
    }
    
    # Sort by reward score in descending order
    output_data["train"].sort(key=lambda x: x["reward_score"], reverse=True)
    
    # Calculate statistics
    scores_array = [x["reward_score"] for x in output_data["train"]]
    stats = {
        "mean_score": sum(scores_array) / len(scores_array),
        "max_score": max(scores_array),
        "min_score": min(scores_array),
        "num_pairs": len(scores_array)
    }
    
    output_data["statistics"] = stats
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Evaluate QA pairs using NVIDIA Reward model')
    parser.add_argument('--input', default='data/synthetic_qa_pairs.json',
                      help='Input file containing QA pairs')
    parser.add_argument('--output', default='data/output/evaluated_qa_pairs.json',
                      help='Output file for evaluated pairs')
    parser.add_argument('--model', 
                      default="nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
                      help='Reward model to use')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize evaluator
    evaluator = QAEvaluator(model_id=args.model)

    # Load QA pairs
    print(f"Loading QA pairs from {args.input}")
    qa_pairs = load_qa_pairs(args.input)

    # Evaluate pairs
    print("Evaluating QA pairs...")
    scores = []
    for pair in tqdm(qa_pairs):
        score = evaluator.evaluate_qa_pair(pair["prompt"], pair["response"])
        scores.append(score)

    # Save results
    print(f"Saving evaluated pairs to {args.output}")
    save_evaluated_pairs(qa_pairs, scores, args.output)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total pairs evaluated: {len(scores)}")
    print(f"Average score: {sum(scores) / len(scores):.3f}")
    print(f"Max score: {max(scores):.3f}")
    print(f"Min score: {min(scores):.3f}")

if __name__ == "__main__":
    main() 