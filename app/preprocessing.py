import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/amazon.csv")

    # Basic cleaning
    df = df.dropna()

    df = df.reset_index(drop=True)


    # Create user_id if not present
    if 'user_id' not in df.columns:
        df['user_id'] = df.index

    # Create product_id if not present
    if 'product_id' not in df.columns:
        df['product_id'] = df.index

    return df