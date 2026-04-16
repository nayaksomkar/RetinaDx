"""
Machine Learning training pipeline using deep learning feature extraction.
"""
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import time
import numpy as np

from src.metrics import calculate_metrics
from src.data_utils import FeatureExtractor, get_image_transform
from src.reports import (
    create_report_directories, create_confusion_matrix_plot,
    create_class_comparison_plot, save_training_history_plot,
    save_json_report, save_text_report, generate_pdf_report
)


def extract_features(dataset_path, transform, device):
    """Extract deep features from dataset."""
    feature_extractor = FeatureExtractor(device)
    from torchvision import datasets
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    X, y = feature_extractor.extract_features(loader, device)
    return X, y, dataset.classes


def train_ml_model(model_name, dataset_path, device):
    """Train a machine learning model on extracted features.
    
    Args:
        model_name: Name of the ML model (knn, randomforest, decisiontree)
        dataset_path: Path to the dataset
        device: Device to use for feature extraction
    """
    print(f"\n{'='*60}\nStarting training for {model_name.upper()}\n{'='*60}")
    
    transform = get_image_transform()
    X, y, class_names = extract_features(dataset_path, transform, device)
    
    print(f"Feature extraction complete! Shape: {X.shape}, Classes: {class_names}")
    print_class_distribution(y, class_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if model_name.lower() == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name.lower() == 'randomforest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name.lower() == 'decisiontree':
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Model must be: knn, randomforest, decisiontree")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_pred, class_names)
    val_metrics = calculate_metrics(y_val, val_pred, class_names)
    test_metrics = calculate_metrics(y_test, test_pred, class_names)

    print("\nPerformance Metrics")
    print(f" Best Validation Accuracy: {val_metrics['macro_avg']['accuracy']:.2f}%")
    print(f" Final Test Accuracy: {test_metrics['macro_avg']['accuracy']:.2f}%")
    print("\nPer-Class Performance Metrics on Test Set:")
    print_metrics(test_metrics)

    dirs = create_report_directories(model_name)
    
    joblib.dump(model, os.path.join(dirs['model_specific_dir'], f"{model_name}_model.joblib"))
    joblib.dump(scaler, os.path.join(dirs['model_specific_dir'], f"{model_name}_scaler.joblib"))

    cm_plot_path = create_confusion_matrix_plot(y_test, test_pred, class_names, dirs, model_name)
    plot_paths = {'confusion_matrix': cm_plot_path}
    plot_paths['class_comparison'] = create_class_comparison_plot(test_metrics, dirs, model_name)

    ml_history = {
        "train_loss": [0.0], "val_loss": [0.0],
        "train_accuracy": [train_metrics['macro_avg']['accuracy']],
        "val_accuracy": [val_metrics['macro_avg']['accuracy']],
    }
    plot_paths['history'] = save_training_history_plot(ml_history, dirs, model_name)

    final_metrics = {
        'best_val_accuracy': val_metrics['macro_avg']['accuracy'],
        'final_test_accuracy': test_metrics['macro_avg']['accuracy'],
        'per_class_metrics': test_metrics,
        'training_history': ml_history
    }
    
    model_info = {
        'architecture': model_name,
        'num_classes': len(class_names),
        'class_names': list(class_names),
        'total_parameters': 'N/A',
        'trainable_parameters': 'N/A'
    }
    
    training_params = {
        'model_name': model_name,
        'training_time': f"{training_time:.2f} seconds",
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    }

    json_path = save_json_report(final_metrics, model_info, training_params, dirs, model_name)
    text_path = save_text_report(final_metrics, model_info, training_params, dirs, model_name)
    pdf_path = generate_pdf_report(final_metrics, model_info, training_params, plot_paths, dirs, model_name)

    print("\n" + "="*60)
    print("Model Information")
    for k, v in model_info.items():
        print(f"  {k}: {v}")
    print("\nTraining Parameters")
    for k, v in training_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest Validation Accuracy: {val_metrics['macro_avg']['accuracy']:.2f}%")
    print(f"Final Test Accuracy: {test_metrics['macro_avg']['accuracy']:.2f}%")
    print("\nGenerated:")
    print(f" JSON: {json_path}")
    print(f" TXT: {text_path}")
    print(f" PDF: {pdf_path}")
    print("="*60 + "\n")

    return model, final_metrics


def print_class_distribution(y, class_names):
    """Print class distribution."""
    labels, counts = np.unique(y, return_counts=True)
    print(f"\n{'-'*38}")
    print(f"{'Class':<18}{'Count':>10}{'Pct':>8}")
    print(f"{'-'*38}")
    total = len(y)
    for idx, count in zip(labels, counts):
        print(f"{class_names[idx]:<18}{count:>10}{(100*count/total):>8.2f}%")
    print(f"{'-'*38}")


def print_metrics(metrics):
    """Print formatted metrics."""
    print(f"{'Class':<18}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'Spec':>8}{'F1':>8}{'FPR':>8}{'FNR':>8}")
    print("-"*74)
    for cname, vals in metrics.items():
        if cname == 'macro_avg':
            continue
        print(f"{cname:<18}{vals['accuracy']:>8.2f}{vals['precision']:>8.2f}{vals['recall']:>8.2f}{vals['specificity']:>8.2f}{vals['f1_score']:>8.2f}{vals['fpr']:>8.2f}{vals['fnr']:>8.2f}")
    macro = metrics['macro_avg']
    print("-"*74)
    print(f"{'Macro avg':<18}{macro['accuracy']:>8.2f}{macro['precision']:>8.2f}{macro['recall']:>8.2f}{macro['specificity']:>8.2f}{macro['f1_score']:>8.2f}{macro['fpr']:>8.2f}{macro['fnr']:>8.2f}")
    print("-"*74)


if __name__ == "__main__":
    DATASET_PATH = r"path\to\your\dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = ['knn', 'randomforest', 'decisiontree']
    models = ['knn']
    
    all_results = {}
    for m in models:
        model, metrics = train_ml_model(m, DATASET_PATH, device)
        all_results[m] = metrics
    
    print("-"*60)
    print("Summary")
    print("-"*60)
    for m in models:
        ma = all_results[m]['per_class_metrics']['macro_avg']
        print(f"{m.upper()}: Test Accuracy={ma['accuracy']:.2f}%, F1={ma['f1_score']:.2f}%")