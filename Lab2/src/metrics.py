import warnings
import numpy as np


def confusion_matrix(y_true: np.ndarray, scores: np.ndarray, t: float) -> tuple:
    pred = scores >= t
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    c = 10 * fn + fp
    return tp, fp, fn, tn, c


def precision_recall(tp: int, fp: int, fn: int) -> tuple:
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec  = tp / (tp + fn) if (tp + fn) > 0 else None
    if prec is None:
        warnings.warn("precision невизначена: жодного прогнозу Fraud (TP+FP=0)")
    if rec is None:
        warnings.warn("recall невизначена: жодного справжнього Fraud (TP+FN=0)")
    return prec, rec