import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    df_processed = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    return df_processed

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def main():
    raw_path = "../breast_cancer_raw.csv"
    processed_path = "breast_cancer_preprocessing.csv"
    
    df = load_raw_data(raw_path)
    df_processed = preprocess_data(df)
    save_processed_data(df_processed, processed_path)

if __name__ == "__main__":
    main()
