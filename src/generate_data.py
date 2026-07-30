import pandas as pd
import numpy as np
import datetime
import os

def generate_mandi_data(start_date="2024-01-01", end_date="2026-07-30", output_file="data/raw/mandi_prices.csv"):
    np.random.seed(42)
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate exact number of days to hit July 30, 2026
    days = (end - start).days + 1 
    dates = [start + datetime.timedelta(days=i) for i in range(days)]
    
    commodities = {
        'Tomato': {'category': 'Vegetable', 'base_price': 30, 'volatility': 12},
        'Potato': {'category': 'Vegetable', 'base_price': 20, 'volatility': 4},
        'Onion': {'category': 'Vegetable', 'base_price': 25, 'volatility': 8},
        'Apple': {'category': 'Fruit', 'base_price': 100, 'volatility': 20},
        'Wheat': {'category': 'Grain', 'base_price': 28, 'volatility': 2},
        'Rice': {'category': 'Grain', 'base_price': 40, 'volatility': 3}
    }
    
    locations = ['Central Mandi', 'North Mandi', 'South Mandi']
    data = []

    for date in dates:
        month = date.month
        seasonal_factor = 1.0 + 0.25 * np.sin(2 * np.pi * month / 12)
        
        for comm, details in commodities.items():
            for loc in locations:
                noise = np.random.normal(0, details['volatility'])
                rainfall_mm = max(0, np.random.exponential(scale=5) if month in [6, 7, 8, 9] else np.random.exponential(scale=1))
                
                price = details['base_price'] * seasonal_factor + noise + (rainfall_mm * 0.1)
                modal_price = max(5.0, round(price, 2))
                min_price = max(4.0, round(modal_price * 0.9, 2))
                max_price = round(modal_price * 1.1, 2)
                
                data.append({
                    'date': date.strftime("%Y-%m-%d"),
                    'market_location': loc,
                    'category': details['category'],
                    'commodity': comm,
                    'min_price': min_price,
                    'max_price': max_price,
                    'modal_price': modal_price,
                    'rainfall_mm': round(rainfall_mm, 2)
                })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"✅ Successfully generated historical price records up to {end_date} -> {output_file}")

if __name__ == "__main__":
    generate_mandi_data()