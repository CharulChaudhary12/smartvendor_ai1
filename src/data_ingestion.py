import pandas as pd
import numpy as np

def preprocess_mandi_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle missing values
    df['modal_price'] = df['modal_price'].ffill()
    
    # Extract temporal features for time-series ML
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df