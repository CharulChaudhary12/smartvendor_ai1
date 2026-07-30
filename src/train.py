import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate(input_file="data/processed/mandi_prices_cleaned.csv", model_dir="models/"):
    print("🚀 Starting Model Training Pipeline...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Processed file not found at {input_file}. Run Step 2 (src/preprocess.py) first.")

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # Define Feature Sets & Target
    features = [
        'month', 'day_of_week', 'day_of_month', 'is_weekend', 'rainfall_mm',
        'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_lag_14',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'price_spread'
    ]
    target = 'modal_price'

    # One-hot encoding for categorical variables (commodity & market_location)
    df_encoded = pd.get_dummies(df, columns=['commodity', 'market_location'], drop_first=True)
    
    # Update feature list with encoded categorical columns
    encoded_features = [col for col in df_encoded.columns if col not in ['date', 'min_price', 'max_price', 'modal_price', 'category', 'year']]

    # Temporal Train-Test Split (Last 20% of timeline for validation/testing)
    df_encoded = df_encoded.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df_encoded) * 0.8)

    X_train = df_encoded.loc[:split_idx, encoded_features]
    y_train = df_encoded.loc[:split_idx, target]
    X_test = df_encoded.loc[split_idx:, encoded_features]
    y_test = df_encoded.loc[split_idx:, target]

    print(f"📊 Dataset Split: {len(X_train)} Training samples | {len(X_test)} Testing samples\n")

    # Initialize Models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    }

    best_model = None
    best_score = float('inf')
    best_model_name = ""

    # Train and Evaluate Models
    results = {}
    for name, model in models.items():
        print(f"⏳ Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"  └─ {name} Metrics -> MAE: ₹{mae:.2f}/kg | RMSE: ₹{rmse:.2f} | R² Score: {r2:.4f}")

        # Keep track of best model based on MAE
        if mae < best_score:
            best_score = mae
            best_model = model
            best_model_name = name

    print(f"\n🏆 Best Model Selected: {best_model_name} (MAE: ₹{best_score:.2f}/kg)")

# Save trained model and feature list artifact
    model_dir = os.path.abspath(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_filepath = os.path.join(model_dir, "smart_vendor_model.pkl")
    features_filepath = os.path.join(model_dir, "model_features.pkl")

    joblib.dump(best_model, model_filepath)
    joblib.dump(encoded_features, features_filepath)

    print(f"💾 Model saved successfully to: {model_filepath}")
    print(f"💾 Feature mappings saved to: {features_filepath}")

if __name__ == "__main__":
    train_and_evaluate()