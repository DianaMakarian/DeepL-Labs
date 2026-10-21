from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from src.metrics import confusion_matrix, precision_recall


def plot_pr_roc(y_val: np.ndarray, scores_val: np.ndarray, out_dir: Path) -> float:
    auc = float(roc_auc_score(y_val, scores_val))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    prec, rec, _ = precision_recall_curve(y_val, scores_val)
    axes[0].plot(rec, prec, color="blue", lw=2)
    axes[0].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall Curve (Validation)")
    axes[0].grid(True)

    fpr, tpr, _ = roc_curve(y_val, scores_val)
    axes[1].plot(fpr, tpr, color="red", lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve (Validation)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "pr_roc_curves.png", dpi=300)
    plt.close()
    return auc


def compare_three_thresholds(y: np.ndarray, scores: np.ndarray, best_t: float) -> pd.DataFrame:
    rows = []
    for label, t in [("Стандартний (0.5)", 0.5), (f"Оптимальний (t*={best_t:.4f})", best_t), ("Усі негативні (+inf)", np.inf)]:
        tp, fp, fn, tn, cost = confusion_matrix(y, scores, t)
        rows.append({"Правило": label, "Поріг": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost})
    return pd.DataFrame(rows)


def test_metrics(y: np.ndarray, scores: np.ndarray, best_t: float) -> dict:
    tp, fp, fn, tn, cost = confusion_matrix(y, scores, best_t)
    prec, rec = precision_recall(tp, fp, fn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost, "Precision": prec, "Recall": rec}


def error_table(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                row_ids: np.ndarray, best_t: float, top_n: int = 5) -> pd.DataFrame:
    pred = scores >= best_t
    rows = []
    for idx in np.flatnonzero((y == 0) & pred)[:top_n]:
        rows.append({"CSV_Row": int(row_ids[idx]), "Type": "FP", "Label": 0,
                     "Score": round(float(scores[idx]), 6),
                     "Dist_to_t*": round(float(scores[idx] - best_t), 6),
                     "Amount": float(amounts[idx])})
    for idx in np.flatnonzero((y == 1) & ~pred)[:top_n]:
        rows.append({"CSV_Row": int(row_ids[idx]), "Type": "FN", "Label": 1,
                     "Score": round(float(scores[idx]), 6),
                     "Dist_to_t*": round(float(scores[idx] - best_t), 6),
                     "Amount": float(amounts[idx])})
    df = pd.DataFrame(rows)
    return df.sort_values("CSV_Row").reset_index(drop=True) if not df.empty else df