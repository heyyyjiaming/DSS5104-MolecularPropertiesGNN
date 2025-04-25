from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def compute_classification_report(y_true, y_probs, threshold=0.5):
    num_tasks = y_true.shape[1]
    metrics = {
        "task": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
        "positives": [],
        "predicted_positives": [],
        "total_num": []
    }

    for i in range(num_tasks):
        mask = ~np.isnan(y_true[:, i])
        y_t = y_true[mask, i]
        y_p = y_probs[mask, i]
        
        y_pred = (y_p >= threshold).astype(int)

        if len(np.unique(y_t)) < 2:
            # Skip if only one class present
            metrics["task"].append(f"task_{i}")
            metrics["precision"].append(np.nan)
            metrics["recall"].append(np.nan)
            metrics["f1"].append(np.nan)
            metrics["auc"].append(np.nan)
            metrics["positives"].append(int(y_t.sum()))
            metrics["predicted_positives"].append(int(y_pred.sum()))
            metrics["total_num"].append(int(len(y_t)))
            continue


        metrics["task"].append(f"task_{i}")
        metrics["precision"].append(precision_score(y_t, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_t, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_t, y_pred, zero_division=0))
        metrics["auc"].append(roc_auc_score(y_t, y_p))
        metrics["positives"].append(int(y_t.sum()))
        metrics["predicted_positives"].append(int(y_pred.sum()))
        metrics["total_num"].append(int(len(y_t)))
        
    # print average metrics
    avg_precision = np.nanmean(metrics["precision"])
    avg_recall = np.nanmean(metrics["recall"])
    avg_f1 = np.nanmean(metrics["f1"])
    avg_auc = np.nanmean(metrics["auc"])
    print(f" - Precision: {avg_precision:.4f}")
    print(f" - Recall: {avg_recall:.4f}")
    print(f" - F1: {avg_f1:.4f}")
    print(f" - AUC: {avg_auc:.4f}")
        

    return pd.DataFrame(metrics)