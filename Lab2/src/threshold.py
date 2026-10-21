import numpy as np
import pandas as pd
from src.metrics import confusion_matrix


def threshold_search_slow(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    candidates = np.append(np.unique(scores), np.inf)

    rows = []
    for t in candidates:
        tp, fp, fn, tn, cost = confusion_matrix(y_true, scores, t)
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost})

    df = pd.DataFrame(rows)
    best_t = df.loc[df["C"] == df["C"].min(), "threshold"].max()
    return float(best_t), df


def threshold_search_fast(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    order = np.argsort(-scores, kind="stable")
    s_sort = scores[order]
    y_sort = y_true[order]

    tp_cum = np.cumsum(y_sort)
    fp_cum = np.cumsum(1 - y_sort)

    group_ends = np.flatnonzero(np.append(s_sort[:-1] != s_sort[1:], True))

    thresholds = s_sort[group_ends]
    tp = tp_cum[group_ends].astype(int)
    fp = fp_cum[group_ends].astype(int)
    fn = (n_pos - tp).astype(int)
    tn = (n_neg - fp).astype(int)
    costs = 10 * fn + fp

    all_t   = np.concatenate([[np.inf], thresholds])
    all_tp  = np.concatenate([[0],      tp])
    all_fp  = np.concatenate([[0],      fp])
    all_fn  = np.concatenate([[n_pos],  fn])
    all_tn  = np.concatenate([[n_neg],  tn])
    all_c   = np.concatenate([[10 * n_pos], costs])

    df = pd.DataFrame({"threshold": all_t, "TP": all_tp, "FP": all_fp,
                        "FN": all_fn, "TN": all_tn, "C": all_c})

    best_idx = int(np.argmin(all_c))
    best_t   = float(all_t[best_idx])
    return best_t, df


def verify_algorithms(y_true: np.ndarray, scores: np.ndarray, label: str = "") -> None:
    t_slow, df_slow = threshold_search_slow(y_true, scores)
    t_fast, df_fast = threshold_search_fast(y_true, scores)

    assert t_slow == t_fast, (f"[{label}] Пороги розійшлися: slow={t_slow}, fast={t_fast}")

    df_s = df_slow.sort_values("threshold", ascending=False).reset_index(drop=True)
    df_f = df_fast.sort_values("threshold", ascending=False).reset_index(drop=True)
    for col in ["TP", "FP", "FN", "TN", "C"]:
        assert (df_s[col].values == df_f[col].values).all(), (
            f"[{label}] Стовпець '{col}' розійшовся між slow та fast")
        
    print(f"  Набір {label!r}: slow t*={t_slow}, fast t*={t_fast} -> PASSED")