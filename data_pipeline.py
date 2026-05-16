import pandas as pd
from sklearn.model_selection import train_test_split



def load_and_sample(path, n=200_000, seed=13):
    data_frame = pd.read_csv(path)
    return data_frame.sample(n, random_state=seed)

def make_target(df, col="suicidal_thoughts", threshold=4):
    y = (df[col] > threshold).astype(int)
    # dropping "Suicidal Thoughts or Intentions"
    X = df.drop(columns=[col])
    return X, y

def split(X, y, seed=13):
    strat = y.astype(str) + "_" + X["age"].astype(str) + "_" + X["biological_sex"].astype(str)
    X_tr, X_tmp, y_tr, y_tmp, str_tr, str_tmp = train_test_split(X, y, strat, test_size=0.3, stratify=strat, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=str_tmp, random_state=seed)
    return X_tr, X_val, X_test, y_tr, y_val, y_test