import subprocess
import argparse
import json
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run multiple VLM finetuning experiments')
    parser.add_argument('--config', type=str, default="experiments.json",
                      help='Path to experiments configuration file (default: experiments.json)')
    parser.add_argument('--runs_per_config', type=int, default=1,
                      help='Number of runs for each dataset-instruction combination (default: 1)')
    return parser.parse_args()

def load_experiments(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        experiments = json.load(f)
    
    # Validate structure
    if not isinstance(experiments, dict):
        raise ValueError("Experiments config must be a dictionary with datasets as keys")
    
    for dataset, instructions in experiments.items():
        if not isinstance(instructions, list):
            raise ValueError(f"Instructions for dataset {dataset} must be a list")
    
    return experiments

def main():
    args = parse_args()
    
    # Load experiments configuration
    experiments = load_experiments(args.config)
    
    # Calculate total number of experiments
    total_experiments = sum(len(instructions) for instructions in experiments.values()) * args.runs_per_config
    
    print(f"Starting experiments at {datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
    print(f"Loaded {len(experiments)} datasets with a total of {total_experiments} experiments to run")
    
    # Track progress
    current_experiment = 0
    
    for dataset, instructions in experiments.items():
        print(f"\nProcessing dataset: {dataset}")
        
        for instruction in instructions:
            for run in range(args.runs_per_config):
                current_experiment += 1
                print(f"\nExperiment {current_experiment}/{total_experiments}")
                print(f"Dataset: {dataset}")
                print(f"Instruction: {instruction[:100]}...")
                print(f"Run: {run + 1}/{args.runs_per_config}")
                
                # Construct command
                cmd = [
                    "python", "multigpu_finetune_vlm.py",
                    "--data", dataset,
                    "--instruction", instruction
                ]
                
                # Run the experiment
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running experiment: {e}")
                    continue
                
                print(f"Completed experiment {current_experiment}/{total_experiments}")
    
    print(f"\nAll experiments completed at {datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")

if __name__ == "__main__":
    main() 