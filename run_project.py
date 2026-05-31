#!/usr/bin/env python3
"""
Complete project runner for Multidisciplinary Deepfake Detection
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and display results"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable] + command.split()[1:], 
                              capture_output=False, text=True, cwd='.')
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("MULTIDISCIPLINARY DEEPFAKE DETECTION - FULL PIPELINE")
    print("="*60)
    
    # Ensure directories exist
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run pipeline
    steps = [
        ("python main.py", "Initial Setup & Basic Model"),
        ("python train_models.py", "Training All Models"),
        ("python evaluate_models.py", "Model Evaluation")
    ]
    
    success_count = 0
    for command, description in steps:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"Failed: {description}")
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {success_count}/{len(steps)} steps successful")
    print(f"{'='*60}")
    
    if success_count == len(steps):
        print("[SUCCESS] Project running successfully!")
        print("\nTrained models available in: models/saved_models/")
        print("- svm_model.pkl")
        print("- randomforest_model.pkl") 
        print("- logisticregression_model.pkl")
    else:
        print("[ERROR] Some steps failed. Check output above.")

if __name__ == "__main__":
    main()