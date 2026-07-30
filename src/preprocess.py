import pandas as pd
import numpy as np
import os

def preprocess_and_engineer_features(
    input_file="data/raw/mandi_prices.csv",
    output_file="data/processed/mandi_prices_cleaned.csv"
):
    print("Starting Data Preprocessing & Feature Engineering...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}. Please run 'python src/generate_data.py' first.")
    
    df = pd.read_csv(input_file)
    
    # 1. Clean Column Names & Format Dates
    df.columns = df.columns.str.lower().str.strip()
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort chronologically for time-series feature creation
    df = df.sort_values(by=['commodity', 'market_location', 'date']).reset_index(drop=True)
    
    # 2. Extract Calendar & Seasonal Features (FR-3)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 3. Time-Series Lag Features (Per Commodity & Location)
    # Allows ML models to look back at past prices to forecast future trends
    for lag in [1, 3, 7, 14]:
        df[f'price_lag_{lag}'] = df.groupby(['commodity', 'market_location'])['modal_price'].shift(lag)
        
    # 4. Rolling Moving Averages & Volatility (7-day & 30-day)
    df['rolling_mean_7'] = df.groupby(['commodity', 'market_location'])['modal_price'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    df['rolling_std_7'] = df.groupby(['commodity', 'market_location'])['modal_price'].transform(
        lambda x: x.shift(1).rolling(window=7).std()
    )
    df['rolling_mean_30'] = df.groupby(['commodity', 'market_location'])['modal_price'].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )
    
    # 5. Price Spread Feature
    df['price_spread'] = df['max_price'] - df['min_price']
    
    # Drop initial rows created with NaN values due to lag/rolling calculations
    initial_count = len(df)
    df_clean = df.dropna().copy()
    dropped_count = initial_count - len(df_clean)
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    
    print(f"Preprocessing complete:")
    print(f" - Rows before clean: {initial_count}")
    print(f" - Rows saved after lag creation: {len(df_clean)} (Dropped {dropped_count} early warm-up rows)")
    print(f" - Processed data saved to: {output_file}")

if __name__ == "__main__":
    preprocess_and_engineer_features()