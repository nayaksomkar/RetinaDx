"""
RetinaDx - Main Entry Point
Retinal Disease Classification System

Usage:
    python main.py dl --model densenet121 --data path/to/dataset
    python main.py ml --model knn --data path/to/dataset
"""
import argparse
import sys
import torch
import numpy as np

from src import train_dl, train_ml


def main():
    parser = argparse.ArgumentParser(description="RetinaDx")
    parser.add_argument("mode", choices=["dl", "ml"], help="Training mode")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--data", type=str, required=True, help="Dataset path")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs (DL only)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.mode == "dl":
        print(f"\nTraining: {args.model}")
        train_dl.train_model(args.model, args.epochs, args.data, args.batch_size)
    else:
        print(f"\nTraining: {args.model}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ml.train_ml_model(args.model, args.data, device)


if __name__ == "__main__":
    main()