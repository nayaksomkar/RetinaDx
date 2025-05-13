import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fpdf import FPDF
from datetime import datetime
from sklearn.metrics import confusion_matrix

REPORT_METRICS = [
    'accuracy', 'precision', 'recall',      # TPR
    'specificity',                          # TNR
    'f1_score', 'fpr', 'fnr'
]

def create_report_directories(model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    model_specific_dir = os.path.join(models_dir, model_name)
    reports_dir = os.path.join(base_dir, "reports")
    model_reports_dir = os.path.join(reports_dir, model_name)
    images_dir = os.path.join(model_reports_dir, "images")
    for d in [models_dir, model_specific_dir, reports_dir, model_reports_dir, images_dir]:
        os.makedirs(d, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return {
        'parent_dir': base_dir,
        'models_dir': models_dir,
        'model_specific_dir': model_specific_dir,
        'reports_dir': reports_dir,
        'model_reports_dir': model_reports_dir,
        'images_dir': images_dir,
        'timestamp': timestamp
    }

def save_training_history_plot(history, dirs, model_name):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label="Training Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.legend()
    plt.title(f"{model_name} Loss [Epoch]")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label="Training Accuracy")
    plt.plot(history['val_accuracy'], label="Validation Accuracy")
    plt.legend()
    plt.title(f"{model_name} Accuracy [Epoch]")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()

    out_path = os.path.join(dirs['images_dir'], f"{model_name}_training_history.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def create_confusion_matrix_plot(y_true, y_pred, class_names, dirs, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    out_path = os.path.join(dirs["images_dir"], f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def create_class_comparison_plot(per_class_metrics, dirs, model_name):
    metrics = REPORT_METRICS
    classes = [k for k in per_class_metrics if k != "macro_avg"]
    data = {metric: [per_class_metrics[c].get(metric, 0) for c in classes] for metric in metrics}
    x = np.arange(len(classes))
    width = 0.12
    plt.figure(figsize=(15, 7))
    for i, metric in enumerate(metrics):
        plt.bar(x + width*i, data[metric], width, label=metric.capitalize())
    plt.xlabel('Class')
    plt.ylabel('Value (%)')
    plt.title(f'{model_name} Per-Class Metrics')
    plt.xticks(x + width*3, classes)
    plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.13))
    plt.tight_layout()
    out_path = os.path.join(dirs['images_dir'], f"{model_name}_class_metrics.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def ensure_macro_avg(per_class_metrics):
    if 'macro_avg' not in per_class_metrics:
        classes = [c for c in per_class_metrics if c != 'macro_avg']
        macro = {}
        for metric in REPORT_METRICS:
            macro[metric] = np.mean([per_class_metrics[c].get(metric, 0) for c in classes])
        per_class_metrics['macro_avg'] = macro
    return per_class_metrics

def save_json_report(metrics, model_info, training_params, dirs, model_name):
    report_data = {
        "timestamp": dirs['timestamp'],
        "model_info": model_info,
        "training_params": training_params,
        "metrics": metrics
    }
    json_path = os.path.join(dirs['model_reports_dir'], f"{model_name}_data.json")
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    return json_path

def save_text_report(metrics, model_info, training_params, dirs, model_name):
    text_path = os.path.join(dirs['model_reports_dir'], f"{model_name}_report.txt")
    per_class_metrics = metrics['per_class_metrics']
    per_class_metrics = ensure_macro_avg(per_class_metrics)
    with open(text_path, 'w') as f:
        f.write(f"=== {model_name.upper()} Model Report ===\n\n")
        f.write(f"Generated on: {dirs['timestamp']}\n\n")
        f.write("Model Information:\n")
        for key, value in model_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTraining Parameters:\n")
        for key, value in training_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n== Macro Average Metrics ==\n")
        macro = per_class_metrics['macro_avg']
        for metric in REPORT_METRICS:
            f.write(f"  {metric.capitalize()}: {macro[metric]:.2f}%\n")
        f.write("\n== Per-Class Metrics ==\n")
        for class_name in [c for c in per_class_metrics if c != 'macro_avg']:
            f.write(f"  {class_name}:\n")
            for metric in REPORT_METRICS:
                f.write(f"    {metric.capitalize()}: {per_class_metrics[class_name][metric]:.2f}%\n")
        f.write("\n")
    return text_path

def generate_pdf_report(metrics, model_info, training_params, plot_paths, dirs, model_name):
    per_class_metrics = metrics['per_class_metrics']
    per_class_metrics = ensure_macro_avg(per_class_metrics)
    macro = per_class_metrics['macro_avg']
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        # Title Page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'{model_name.upper()} Model Report', ln=True, align='C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f"Generated on: {dirs['timestamp']}", ln=True)

        # Model Info
        pdf.ln(8)
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 10, 'Model Information', ln=True)
        pdf.set_font('Arial', '', 11)
        for key, value in model_info.items():
            pdf.cell(0, 8, f"{key}: {value}", ln=True)

        pdf.ln(8)
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 10, 'Training Parameters', ln=True)
        pdf.set_font('Arial', '', 11)
        for key, value in training_params.items():
            pdf.cell(0, 8, f"{key}: {value}", ln=True)

        # Macro Average
        pdf.add_page()
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 10, 'Macro Average Performance', ln=True)
        pdf.set_font('Arial', '', 11)
        for metric in REPORT_METRICS:
            pdf.cell(0, 8, f"{metric.capitalize()}: {macro[metric]:.2f}%", ln=True)

        # Plots
        for label, path in plot_paths.items():
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            plot_title = label.replace('_', ' ').title()
            pdf.cell(0, 10, plot_title, ln=True)
            pdf.image(path, x=10, y=None, w=190)

        # Per-class table
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Per-Class Metrics Table", ln=True)
        pdf.ln(6)
        pdf.set_font('Arial', 'B', 9)
        cellw = 24
        pdf.cell(cellw, 8, "Class", 1)
        for m in REPORT_METRICS:
            pdf.cell(cellw, 8, m.capitalize(), 1)
        pdf.ln()
        pdf.set_font('Arial', '', 9)

        for cname in [c for c in per_class_metrics if c != 'macro_avg']:
            pdf.cell(cellw, 8, str(cname), 1)
            for m in REPORT_METRICS:
                pdf.cell(cellw, 8, f"{per_class_metrics[cname][m]:.2f}", 1)
            pdf.ln()
        # Macro avg
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(cellw, 8, "Macro Avg", 1)
        for m in REPORT_METRICS:
            pdf.cell(cellw, 8, f"{macro[m]:.2f}", 1)
        pdf.ln()

        # Save
        pdf_path = os.path.join(dirs['model_reports_dir'], f"{model_name}_report.pdf")
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return None