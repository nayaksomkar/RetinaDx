"""
Configuration settings for RetinaDx.
"""

# Dataset path - change this to your dataset location
DATASET_PATH = "RetinaDxDataSet/Dataset X"

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

# Model settings
DL_MODELS = ["densenet121", "resnet50", "resnet101", "xception"]
ML_MODELS = ["knn", "randomforest", "decisiontree"]

# Hardware
DEVICE = "cuda"  # or "cpu"
NUM_WORKERS = 4

# Report settings
REPORT_DIR = "reports"
MODEL_DIR = "models"