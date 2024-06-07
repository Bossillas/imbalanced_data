import yaml
from sklearn.model_selection import train_test_spli
import data_generation as dg
import model as m

# read config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
def pipeline(cfg: dict):
    
    # generate data
    df = dg.data_generation(cfg["data"]["ratio"],
                            cfg["data"]["n_samples"],
                            cfg["data"]["n_features"],
                            cfg["data"]["n_informative"],
                            cfg["data"]["random_state"],
                            cfg["data"]["target_randomness"])
    
    df = dg.fix_imbalance(df, 
                          cfg["data"]["fix_imbalance_strategy"],
                          cfg["data"]["random_state"])
    
    # fit the models
    lr = m.fit_model(df.drop(columns=["target"]),
                     df["target"],
                     cfg["model"]["logistic"],
                     model="logistic")
    xgb = m.fit_model(df.drop(columns=["target"]),
                      df["target"],
                      cfg["model"]["xgboost"],
                      model="xgboost")


if __name__ == "__main__":
    pipeline(config)