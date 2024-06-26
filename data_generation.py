import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def data_generation(
    ratio: float,
    n_samples: int,
    n_features: int = 20,
    n_informative: int = 7,
    random_state: int = 7,
    target_randomness: float = 0.05
) -> pd.DataFrame:
    if ratio <= 0 or ratio >= 1:
        raise ValueError("ratio but be between 0 and 1")
    
    # generate binary classification data
    X, y = make_classification(weights=[1-ratio, ratio],
                               n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               random_state=random_state,
                               flip_y=target_randomness)
    
    # convert data into dataframe format
    df = pd.DataFrame(X, columns=[f"feat_{i + 1}" for i in range(n_features)])
    df["target"] = y
    
    return df

def fix_imbalance(
    df: pd.DataFrame,
    strategy: str = "nothing",
    random_state: int = 7
) -> pd.DataFrame:
    
    if strategy == "nothing":
        return df
    elif strategy == "upsample":
        df0 = df[df["target"] == 0]
        df1 = df[df["target"] == 1]
        if df0.shape[0] > df1.shape[0]:
            df1.sample(n=df0.shape[0], random_state=random_state)
        elif df0.shape[0] < df1.shape[0]:
            df0.sample(n=df1.shape[0], random_state=random_state)
        else:
            return df
        df = pd.concat([df0, df1])\
            .sample(frac=1, random_state=random_state)\
            .reset_index(drop=True)    
        return df
    elif strategy == "downsample":
        df0 = df[df["target"] == 0]
        df1 = df[df["target"] == 1]
        if df0.shape[0] > df1.shape[0]:
            df0.sample(n=df1.shape[0], random_state=random_state)
        elif df0.shape[0] < df1.shape[0]:
            df1.sample(n=df0.shape[0], random_state=random_state)
        else:
            return df
        df = pd.concat([df0, df1])\
            .sample(frac=1, random_state=random_state)\
            .reset_index(drop=True)    
        return df
    else:
        raise ValueError(f"strategy argument must be one of the following: 'nothing', 'upsample', or 'downsample' but recieved '{strategy}'")