"""
RetinaDx - Retinal Disease Classification System
Main package initialization.
"""

# Metrics
from src.metrics import calculate_metrics

# Data utilities
from src.data_utils import (
    FeatureExtractor,
    gpu_check,
    get_vanilla_transform,
    get_advanced_transform,
    get_image_transform,
    load_dataset,
)

# Reports
from src.reports import (
    create_report_directories,
    save_training_history_plot,
    create_confusion_matrix_plot,
    create_class_comparison_plot,
    save_json_report,
    save_text_report,
    generate_pdf_report,
)

# Training pipelines
import src.train_dl
import src.train_ml


__all__ = [
    "calculate_metrics",
    "FeatureExtractor",
    "gpu_check",
    "get_vanilla_transform",
    "get_advanced_transform",
    "get_image_transform",
    "load_dataset",
    "create_report_directories",
    "save_training_history_plot",
    "create_confusion_matrix_plot",
    "create_class_comparison_plot",
    "save_json_report",
    "save_text_report",
    "generate_pdf_report",
]