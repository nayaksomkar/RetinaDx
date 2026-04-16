"""
Deep Learning training pipeline for retinal disease classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import os
import numpy as np
import time

from src.metrics import calculate_metrics
from src.data_utils import get_image_transform, load_dataset
from src.reports import (
    create_report_directories, save_training_history_plot,
    create_confusion_matrix_plot, create_class_comparison_plot,
    save_json_report, save_text_report, generate_pdf_report
)


def get_model(model_name, num_classes):
    """Create model by name."""
    model_name = model_name.lower()
    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "xception":
        model = timm.create_model("xception", pretrained=True)
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Model must be: densenet121, resnet50, resnet101, or xception")
    return model


def print_class_distribution(dataset):
    """Print class distribution from dataset."""
    class_counts = {}
    total_samples = len(dataset)
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\nClass Distribution:")
    print("-" * 60)
    print(f"{'Class Index':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    for class_idx, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{class_idx:<15} {count:<10} {percentage:.2f}%")
    print("-" * 60)
    print(f"Total samples: {total_samples}")
    return class_counts


def print_metrics(metrics, phase):
    """Print metrics to console."""
    print(f"\n{phase} Per-Class Metrics:")
    print("-" * 117)
    print(f"{'Class':<20} {'Acc':>8}  {'Prec':>8}  {'TPR':>8}  {'TNR':>8} "
          f"{'F1':>8}  {'FPR':>8}  {'FNR':>8}")
    print("-" * 117)
    for cname, m in metrics.items():
        if cname == "macro_avg":
            continue
        print(f"{cname:<20} {m['accuracy']:8.2f} {m['precision']:8.2f} {m['recall']:8.2f} "
              f"{m['specificity']:8.2f} {m['f1_score']:8.2f} {m['fpr']:8.2f} {m['fnr']:8.2f}")
    macro = metrics["macro_avg"]
    print("-" * 117)
    print(f"{'Macro AVG':<20} {macro['accuracy']:8.2f} {macro['precision']:8.2f} {macro['recall']:8.2f} "
          f"{macro['specificity']:8.2f} {macro['f1_score']:8.2f} {macro['fpr']:8.2f} {macro['fnr']:8.2f}")
    print("-" * 117)


def train_model(model_name, num_epochs=50, DATASET_PATH="PATH_TO_DATA", batch_size=32):
    """Train a deep learning model.
    
    Args:
        model_name: Name of the model architecture
        num_epochs: Number of training epochs
        DATASET_PATH: Path to the dataset
        batch_size: Batch size for training
    """
    dirs = create_report_directories(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {device}")

    transform = get_image_transform(config.get('transforms.mode', 'advanced'))
    data = load_dataset(DATASET_PATH, transform, batch_size)
    num_classes = data['num_classes']
    class_names = data['classes']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    
    print(f"Classes: {class_names}")
    print_class_distribution(train_loader.dataset)

    model = get_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "learning_rates": [],
    }
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_predictions, train_labels_list = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_predictions.extend(predicted.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                "loss": train_loss / len(train_loader) if len(train_loader) > 0 else 0,
                "acc": 100.0 * train_correct / train_total if train_total > 0 else 0,
            })

        train_accuracy = 100.0 * train_correct / train_total if train_total else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) else 0

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_predictions, val_labels_list = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_accuracy = 100.0 * val_correct / val_total if val_total else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) else 0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        print(f"\nTrain: {train_accuracy:.2f}%  |  Val: {val_accuracy:.2f}%")
        
        print_metrics(
            calculate_metrics(train_labels_list, train_predictions, class_names),
            f"Epoch {epoch+1} Training"
        )
        print_metrics(
            calculate_metrics(val_labels_list, val_predictions, class_names),
            f"Epoch {epoch+1} Validation"
        )

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(dirs["model_specific_dir"], f"best_{model_name}_model.pt")
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        scheduler.step(avg_val_loss)
        if val_accuracy >= 90.0:
            print("Stopping: Validation accuracy above 90%!")
            break

    model.eval()
    test_predictions, test_labels_list = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                outputs = model(images)
            _, predicted = outputs.max(1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_metrics = calculate_metrics(test_labels_list, test_predictions, class_names)
    print_metrics(test_metrics, "TEST")

    gpu_info = {
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB" if torch.cuda.is_available() else "N/A",
    }

    training_params = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "optimizer": optimizer.__class__.__name__,
        "device": str(device),
        "gpu_info": gpu_info,
        "total_training_time": f"{(time.time() - start_time)/60:.2f} minutes",
    }
    
    model_info = {
        "architecture": model_name,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    final_metrics = {
        "best_val_accuracy": best_val_acc,
        "final_test_accuracy": test_metrics['macro_avg']['accuracy'],
        "per_class_metrics": test_metrics,
        "training_history": history,
    }

    plot_paths = {}
    plot_paths["history"] = save_training_history_plot(history, dirs, model_name)
    plot_paths["confusion_matrix"] = create_confusion_matrix_plot(
        test_labels_list, test_predictions, class_names, dirs, model_name
    )
    plot_paths["class_comparison"] = create_class_comparison_plot(
        test_metrics, dirs, model_name
    )

    json_path = save_json_report(final_metrics, model_info, training_params, dirs, model_name)
    text_path = save_text_report(final_metrics, model_info, training_params, dirs, model_name)
    pdf_path = generate_pdf_report(final_metrics, model_info, training_params, plot_paths, dirs, model_name)

    print("\nReports generated:")
    print(f"JSON: {json_path}")
    print(f"TXT: {text_path}")
    print(f"PDF: {pdf_path}")

    if test_metrics["macro_avg"]["accuracy"] < 90.0:
        print("WARNING: Test accuracy below 90%!")

    return model, history, test_metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    DATASET_PATH = r"path\to\your\dataset"
    models_to_train = ['densenet121', 'resnet50', 'resnet101']
    models_to_train = ['densenet121']
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training: {model_name.upper()}")
        train_model(model_name, DATASET_PATH=DATASET_PATH)
        print(f"Finished: {model_name.upper()}")
        print(f"{'='*50}\n")