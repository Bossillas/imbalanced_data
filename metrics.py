import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Tuple, Union


def auc_roc(
    y_pred: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series]
): 
    return roc_auc_score(y_true, y_pred)

def auc_pr(
    y_pred: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series]
):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def capture_rate(
    y_pred: Union[np.ndarray, pd.Series],
    y_true: Union[np.ndarray, pd.Series],
    top_percent: float = 0.3
):
    order = np.argsort(-y_pred)
    sorted_y_true = y_true[order]
    tot_bad = y_true.sum()
    captured_bad = sorted_y_true[:int(len(y_true)*top_percent)]
    return captured_bad / tot_bad