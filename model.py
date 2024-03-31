import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from typing import Tuple, Union


def fit_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    params: dict,
    model: str = "xgboost",
    cv: int = 5,
    scoring: str = "auc"
):
    if model not in ("xgboost", "logistic"):
        raise ValueError(f"model argument must be one of the following: 'xgboost' or 'logistic' but recieved '{model}'")
    elif model == "xgboost":
        model_ = XGBClassifier()
    else:
        model_ = LogisticRegression()
        
    grid_model = GridSearchCV(model_, params, cv=cv, scoring=scoring)
    
    grid_model.fit(X, y)
    
    return grid_model.best_estimator_