"""
Metrics calculation utilities.
"""
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_pred, classes):
    """Calculate detailed metrics for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    metrics = {}
    for i, cname in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        acc = 100. * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        precision = 100. * tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 100. * 2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
        fpr = 100. * fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = 100. * fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics[cname] = dict(
            accuracy=acc,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1_score=f1,
            fpr=fpr,
            fnr=fnr,
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)
        )
    
    macro = {}
    keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'fpr', 'fnr']
    for k in keys:
        macro[k] = np.mean([metrics[c][k] for c in classes])
    metrics['macro_avg'] = macro
    return metrics