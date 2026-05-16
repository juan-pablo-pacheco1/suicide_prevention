import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_sample(path, n=200_000, seed=13):
    data_frame = pd.read_csv(path, low_memory=False)
    return data_frame.sample(n, random_state=seed)

def clean(df):
    df = df.copy().dropna(axis=1, how="all")
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col])[0]
    df = df.fillna(df.median(numeric_only=True))
    return df

def make_target(df, col="suicidal_thoughts", threshold=4):
    y = (df[col] > threshold).astype(int)
    # dropping "Suicidal Thoughts or Intentions"
    X = df.drop(columns=[col])
    return X, y

def split(X, y, seed=13):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_tr, X_val, X_test, y_tr, y_val, y_test