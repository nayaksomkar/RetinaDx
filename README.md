# RetinaDx - Retinal Disease Classification System

A deep learning and machine learning system for classifying retinal diseases from fundus images. Supports multiple model architectures including DenseNet, ResNet, and traditional ML classifiers.

For full documentation, see: [RetinaDxGit.pdf](assets/RetinaDxGit.pdf)

## Features

- **Deep Learning Models**: DenseNet121, ResNet50, ResNet101, Xception
- **Machine Learning Models**: KNN, Random Forest, Decision Tree
- **Automatic Feature Extraction**: Uses pre-trained DenseNet121 for feature extraction
- **Detailed Reporting**: Generates JSON, TXT, and PDF reports with confusion matrices and metrics
- **Multi-class Metrics**: Accuracy, Precision, Recall, Specificity, F1-Score, FPR, FNR

## Dataset Structure

```
dataset/
├── cataract/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── diabetic_retinopathy/
│   ├── image1.jpg
│   └── ...
├── glaucoma/
│   └── ...
└── normal/
```

## Installation

```bash
# Clone the repository
git clone https://github.com/nayaksomkar/RetinaDx.git
cd RetinaDx

# Create virtual environment (option 1: using Python)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

# OR create virtual environment (option 2: using script)
python setup_venv.py

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Deep Learning Training

```bash
# Train with DenseNet121
python main.py dl --model densenet121 --data path/to/dataset --epochs 50

# Train with ResNet50
python main.py dl --model resnet50 --data path/to/dataset

# Train with Xception
python main.py dl --model xception --data path/to/dataset
```

### Machine Learning Training

```bash
# Train with KNN
python main.py ml --model knn --data path/to/dataset

# Train with Random Forest
python main.py ml --model randomforest --data path/to/dataset

# Train with Decision Tree
python main.py ml --model decisiontree --data path/to/dataset
```

## Training Pipeline

### Deep Learning (DL)

The DL pipeline uses transfer learning with pre-trained ImageNet models:

1. **Data Augmentation**: Advanced transforms including random crop, rotation, flip, color jitter, perspective
2. **Model**: Replace final classification layer for desired number of classes
3. **Optimizer**: AdamW with learning rate 0.001, weight decay 0.01
4. **Scheduler**: ReduceLROnPlateau with patience 5
5. **Early Stopping**: Patience of 10 epochs
6. **Metrics**: Per-class accuracy, precision, recall, specificity, F1-score

### Machine Learning (ML)

The ML pipeline extracts deep features first:

1. **Feature Extraction**: Use pre-trained DenseNet121 as feature extractor
2. **Train/Val/Test Split**: 80/10/10% stratified split
3. **Scaling**: StandardScaler for feature normalization
4. **Model Training**: Train KNN, Random Forest, or Decision Tree

## Output Reports

After training, reports are generated in `reports/{model_name}/`:

- `{model_name}_data.json` - Full metrics in JSON format
- `{model_name}_report.txt` - Human-readable text report
- `{model_name}_report.pdf` - PDF report with charts
- `images/{model_name}_confusion_matrix.png` - Confusion matrix plot
- `images/{model_name}_training_history.png` - Training curves
- `images/{model_name}_class_metrics.png` - Per-class metrics bar chart

Model checkpoints saved in `models/{model_name}/`.

## Documentation

For complete details, training results, confusion matrices, and performance metrics, see:
[RetinaDxGit.pdf](assets/RetinaDxGit.pdf)

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision | True positive rate for predictions |
| Recall (TPR) | Sensitivity, hit rate |
| Specificity (TNR) | True negative rate |
| F1-Score | Harmonic mean of precision/recall |
| FPR | False positive rate |
| FNR | False negative rate |

## Project Structure

```
RetinaDx/
├── main.py           # Entry point
├── config.py         # Settings
├── requirements.txt  # Dependencies
├── README.md         # This file
├── .gitignore       # Git ignore
├── src/
│   ├── __init__.py
│   ├── config.py    # Settings
│   ├── metrics.py   # Metrics calculation
│   ├── data_utils.py# Data loading
│   ├── reports.py   # Report generation
│   ├── train_dl.py  # DL training
│   └── train_ml.py  # ML training
├── assets/          # Images
└── RetinaDxDataSet/ # Your dataset
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision
- timm
- scikit-learn
- matplotlib
- seaborn
- fpdf
- tqdm

## Sample Results

Run training to generate results. Example outputs:

- Confusion matrices
- Training history plots  
- Per-class performance charts

After running, check:

- `reports/{model_name}/images/` for generated plots
- `reports/{model_name}/{model_name}_report.pdf` for full PDF report

## Hardware

- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM

## Acknowledgments

Pre-trained models from PyTorch Hub and timm library.