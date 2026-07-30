import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import datetime

# --- PAGE CONFIGURATION & CINEMATIC UI ---
st.set_page_config(page_title="Smart Vendor AI", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for high-contrast, premium dark mode interface
st.markdown("""
    <style>
    .main { background-color: #050505; }
    h1, h2, h3, h4 { color: #FFFFFF; font-family: 'Inter', sans-serif; font-weight: 400; }
    
    /* Cinematic glowing metric cards */
    .metric-card { 
        background: linear-gradient(145deg, #111317, #0B0C10); 
        padding: 25px; 
        border-radius: 12px; 
        border: 1px solid #1F2329;
        box-shadow: 0 8px 32px rgba(0, 240, 255, 0.05);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(0, 240, 255, 0.15);
        border: 1px solid #00F0FF;
    }
    
    .highlight-green { color: #00FF41; font-weight: bold; text-shadow: 0 0 10px rgba(0, 255, 65, 0.5); }
    .highlight-red { color: #FF003C; font-weight: bold; text-shadow: 0 0 10px rgba(255, 0, 60, 0.5); }
    .highlight-blue { color: #00F0FF; font-weight: bold; text-shadow: 0 0 10px rgba(0, 240, 255, 0.5); }
    
    .layman-text { color: #A0AEC0; font-size: 14px; margin-top: 5px; line-height: 1.5; }
    .b2b-card { 
        background-color: #12141A; 
        padding: 20px; 
        border-left: 5px solid #00F0FF; 
        border-radius: 5px; 
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- MOCK DATABASE FOR LOGIN ---
USER_CREDENTIALS = {
    "farmer1": {"password": "password123", "role": "Farmer"},
    "vendor1": {"password": "password123", "role": "Vendor / Shopkeeper"},
    "consumer1": {"password": "password123", "role": "Consumer"},
    "admin1": {"password": "admin123", "role": "Admin"}
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.username = None

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    if not os.path.exists("data/processed/mandi_prices_cleaned.csv"):
        return pd.DataFrame()
    df = pd.read_csv("data/processed/mandi_prices_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model_and_features():
    if not os.path.exists("models/smart_vendor_model.pkl") or not os.path.exists("models/model_features.pkl"):
        return None, None
    return joblib.load("models/smart_vendor_model.pkl"), joblib.load("models/model_features.pkl")

# --- LOGIN INTERFACE ---
def login_page():
    st.markdown("<h1 style='text-align: center; margin-top: 80px;'>🔐 Smart Vendor AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>AI-Powered Market Intelligence & Price Forecasting Terminal</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("System ID")
            password = st.text_input("Passcode", type="password")
            submit = st.form_submit_button("Authenticate")
            
            if submit:
                if username in USER_CREDENTIALS and USER_CREDENTIALS[username]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_role = USER_CREDENTIALS[username]["role"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Authentication Failed. Check your username or password.")
        
        st.info("💡 **Demo Credentials:**\n\n- **Farmer:** `farmer1` / `password123`\n- **Vendor:** `vendor1` / `password123`\n- **Consumer:** `consumer1` / `password123`\n- **Admin:** `admin1` / `admin123`")

# --- MAIN DASHBOARD ---
def main_dashboard():
    df = load_data()
    model, expected_features = load_model_and_features()

    if df.empty or model is None:
        st.error("Dataset or Trained Model not found! Please run step 1 (`generate_data.py`), step 2 (`preprocess.py`), and step 3 (`train.py`) first.")
        st.stop()

    user_role = st.session_state.user_role

    st.title("🌐 Market Intelligence Terminal")
    st.sidebar.markdown(f"### 👤 {st.session_state.username.upper()} | {user_role.upper()}")
    
    if st.sidebar.button("Terminate Session"):
        st.session_state.logged_in = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Select Market Data")
    
    selected_category = st.sidebar.selectbox("Category", df['category'].unique())
    selected_commodity = st.sidebar.selectbox("Commodity", df[df['category'] == selected_category]['commodity'].unique())
    selected_location = st.sidebar.selectbox("Market Location", df['market_location'].unique())

    subset = df[(df['commodity'] == selected_commodity) & (df['market_location'] == selected_location)].sort_values('date')
    latest_row = subset.iloc[-1]
    latest_price = latest_row['modal_price']
    latest_date_str = latest_row['date'].strftime('%B %d, %Y')
    
    st.markdown("---")
    
    # --- SIMPLIFIED, EYE-CATCHING METRICS ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:#888; font-size: 16px;">Today's Price ({latest_date_str})</h3>
            <h1 class="highlight-blue" style="margin: 10px 0;">₹{latest_price:.2f} / kg</h1>
            <p class="layman-text">This is the average trading rate for {selected_commodity} at {selected_location} today.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:#888; font-size: 16px;">Today's Market Range</h3>
            <h1 style="margin: 10px 0; color: #FFF;">₹{latest_row['min_price']:.2f} - ₹{latest_row['max_price']:.2f}</h1>
            <p class="layman-text">The minimum and maximum price bounds observed in the market today.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        market_health = "Stable" if latest_row['rolling_std_7'] < 2 else "Highly Unpredictable"
        health_color = "#00FF41" if market_health == "Stable" else "#FF003C"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:#888; font-size: 16px;">Market Stability</h3>
            <h1 style="margin: 10px 0; color: {health_color}; text-shadow: 0 0 10px {health_color}80;">{market_health}</h1>
            <p class="layman-text">Evaluated from price fluctuation standard deviation over the last 7 days.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SEARCH & VISUALIZATION (LAYMAN FRIENDLY) ---
    col_title, col_search = st.columns([2, 1])
    with col_title:
        st.subheader("🔮 AI Future Price Forecaster")
        st.markdown("<p class='layman-text'>Predict market prices for any future date. The glowing band indicates the predicted high-and-low safety range.</p>", unsafe_allow_html=True)
    
    latest_date_dt = latest_row['date'].date()
    with col_search:
        target_date = st.date_input(
            "📅 Check a specific future date:", 
            value=latest_date_dt + datetime.timedelta(days=7),
            min_value=latest_date_dt + datetime.timedelta(days=1), 
            max_value=latest_date_dt + datetime.timedelta(days=90),
            key="target_date_picker"
        )

    # Generate Future Data up to selected target date
    future_start_date = subset['date'].max() + datetime.timedelta(days=1)
    days_to_predict = max(30, (target_date - latest_date_dt).days + 5)
    future_dates = [future_start_date + datetime.timedelta(days=i) for i in range(days_to_predict)]
    
    future_df = pd.DataFrame({'date': future_dates})
    for col in expected_features: 
        future_df[col] = 0 
    future_df['month'] = future_df['date'].dt.month
    future_df['day_of_week'] = future_df['date'].dt.dayofweek
    future_df['day_of_month'] = future_df['date'].dt.day
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['rainfall_mm'] = latest_row['rainfall_mm']
    future_df['price_lag_1'] = latest_price
    future_df['price_lag_3'] = latest_price
    future_df['price_lag_7'] = latest_price
    future_df['price_lag_14'] = latest_price
    future_df['rolling_mean_7'] = latest_row['rolling_mean_7']
    future_df['rolling_std_7'] = latest_row['rolling_std_7']
    future_df['rolling_mean_30'] = latest_row['rolling_mean_30']
    future_df['price_spread'] = latest_row['price_spread']

    if f'commodity_{selected_commodity}' in future_df.columns: 
        future_df[f'commodity_{selected_commodity}'] = 1
    if f'market_location_{selected_location}' in future_df.columns: 
        future_df[f'market_location_{selected_location}'] = 1
    
    # AI Model Inference
    future_df['predicted_price'] = model.predict(future_df[expected_features])
    volatility_margin = max(1.5, latest_row['rolling_std_7'] * 1.25)
    future_df['upper_bound'] = future_df['predicted_price'] + volatility_margin
    future_df['lower_bound'] = future_df['predicted_price'] - volatility_margin

    target_date_pd = pd.to_datetime(target_date)
    target_prediction = future_df[future_df['date'] == target_date_pd]
    
    if not target_prediction.empty:
        target_base = target_prediction['predicted_price'].values[0]
        target_min = target_prediction['lower_bound'].values[0]
        target_max = target_prediction['upper_bound'].values[0]
        
        st.markdown(f"""
        <div style="background: rgba(0, 240, 255, 0.05); padding: 25px; border-radius: 12px; border: 1px solid #00F0FF; margin-bottom: 30px; box-shadow: 0 0 20px rgba(0, 240, 255, 0.1);">
            <h3 style="margin: 0 0 15px 0; color: #FFF;">🎯 Prediction for {target_date.strftime('%B %d, %Y')}</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 50px;">
                <div>
                    <span style="color: #A0AEC0; font-size: 16px;">Most Likely Estimated Price</span><br>
                    <span class="highlight-blue" style="font-size: 32px;">₹{target_base:.2f} / kg</span>
                </div>
                <div>
                    <span style="color: #A0AEC0; font-size: 16px;">Expected Safe Range (Min - Max)</span><br>
                    <span class="highlight-green" style="font-size: 32px;">₹{target_min:.2f} to ₹{target_max:.2f}</span>
                </div>
            </div>
            <p class="layman-text" style="margin-top: 15px; max-width: 850px;">
                <strong>Understanding this result:</strong> On this date, the AI model projects the market rate for <strong>{selected_commodity}</strong> to be approximately <strong>₹{target_base:.2f}/kg</strong>. Standard market variations suggest it could range between <strong>₹{target_min:.2f}</strong> and <strong>₹{target_max:.2f}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # High-Contrast Plotly Chart
    fig = go.Figure()
    
    # Historical actual prices
    fig.add_trace(go.Scatter(
        x=subset['date'].tail(30), y=subset['modal_price'].tail(30), 
        mode='lines', name='Past Prices (Last 30 Days)', 
        line=dict(color='#00F0FF', width=3),
    ))
    
    # Upper Bound 
    fig.add_trace(go.Scatter(
        x=future_df['date'], y=future_df['upper_bound'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    # Lower Bound with color fill
    trend_color_rgb = '0, 255, 65' if future_df['predicted_price'].iloc[-1] > latest_price else '255, 0, 60'
    fig.add_trace(go.Scatter(
        x=future_df['date'], y=future_df['lower_bound'],
        mode='lines', line=dict(width=0), fill='tonexty', 
        fillcolor=f'rgba({trend_color_rgb}, 0.15)', name='Estimated Safe Range'
    ))

    # Predicted line
    trend_color = '#00FF41' if future_df['predicted_price'].iloc[-1] > latest_price else '#FF003C'
    fig.add_trace(go.Scatter(
        x=future_df['date'], y=future_df['predicted_price'], 
        mode='lines', name='AI Predicted Price', 
        line=dict(color=trend_color, width=3, dash='dot')
    ))

    # Target Date Marker
    if not target_prediction.empty:
        fig.add_trace(go.Scatter(
            x=[target_date_pd], y=[target_base],
            mode='markers', name='Your Searched Date',
            marker=dict(size=14, color='#FFFFFF', line=dict(width=3, color='#00F0FF'))
        ))

    fig.update_layout(
        plot_bgcolor='#050505', paper_bgcolor='#050505',
        font=dict(color='#A0AEC0'), margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor='#1A202C', title="Timeline"), 
        yaxis=dict(showgrid=True, gridcolor='#1A202C', title="Price per KG (₹)"),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, width="stretch")

    # --- DATE-WISE DATA LEDGER (TABULAR FORM) ---
    st.markdown("---")
    st.subheader("📅 Date-Wise Market Ledger")
    st.markdown("<p class='layman-text'>Complete day-by-day record combining past actual prices and future AI forecasts up to your searched date.</p>", unsafe_allow_html=True)
    
    # Combine Historical (last 30 days) and Future data into one table
    hist_table = subset[['date', 'modal_price', 'min_price', 'max_price']].tail(30).copy()
    hist_table['Status'] = 'Historical (Past)'
    hist_table.rename(columns={'modal_price': 'Average Price', 'min_price': 'Lowest Price', 'max_price': 'Highest Price'}, inplace=True)
    
    fut_table = future_df[future_df['date'] <= target_date_pd][['date', 'predicted_price', 'lower_bound', 'upper_bound']].copy()
    fut_table['Status'] = 'AI Prediction (Future)'
    fut_table.rename(columns={'predicted_price': 'Average Price', 'lower_bound': 'Lowest Price', 'upper_bound': 'Highest Price'}, inplace=True)
    
    # Rounding values
    fut_table['Average Price'] = fut_table['Average Price'].round(2)
    fut_table['Lowest Price'] = fut_table['Lowest Price'].round(2)
    fut_table['Highest Price'] = fut_table['Highest Price'].round(2)
    
    combined_ledger = pd.concat([hist_table, fut_table]).reset_index(drop=True)
    combined_ledger['date'] = combined_ledger['date'].dt.strftime('%Y-%m-%d')
    combined_ledger.set_index('date', inplace=True)
    
    st.dataframe(
        combined_ledger.style.map(
            lambda x: 'color: #00F0FF;' if x == 'Historical (Past)' else 'color: #00FF41;' if x == 'AI Prediction (Future)' else '', 
            subset=['Status']
        ),
        width="stretch",
        height=400
    )

    # --- ADVANCED BUSINESS INTEGRATIONS ---
    st.markdown("---")
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("🏢 Virtual Food Campus (B2B Integration)")
        st.write("Connecting local surplus with commercial food-tech platforms.")
        if future_df['predicted_price'].mean() < latest_price:
            st.markdown(f"""
            <div class="b2b-card" style="border-left-color: #00FF41;">
                <h4 style="margin:0; color:#00FF41;">🟢 Bulk Purchase Alert Active</h4>
                <p style="margin:5px 0; color:#aaa;">Projected price drop detected. Initiating API broadcast to the <strong>SwiggyVerse Experience Zone</strong> and local cloud kitchens for bulk procurement of {selected_commodity}.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Broadcast Deal to B2B Buyers"):
                st.success("Deal broadcasted to SwiggyVerse network!")
        else:
            st.markdown(f"""
            <div class="b2b-card" style="border-left-color: #555;">
                <h4 style="margin:0; color:#888;">⚪ Market Stable</h4>
                <p style="margin:5px 0; color:#aaa;">No supply surplus detected. Standard commercial fulfillment rates apply.</p>
            </div>
            """, unsafe_allow_html=True)

    with colB:
        st.subheader("⏱️ Wastage Mitigation Engine")
        st.write("Dynamic pricing curves for perishable inventory.")
        days_to_spoil = st.slider("Days Until Estimated Spoilage", 1, 14, 5)
        
        if days_to_spoil <= 3:
            discount = (4 - days_to_spoil) * 15
            st.markdown(f"""
            <div style="background:#2D1518; padding:15px; border-radius:8px; border: 1px solid #FF003C;">
                <span class="highlight-red">⚠️ HIGH SPOILAGE RISK</span><br>
                <span style="font-size: 24px; color: white;">Suggested Markdown: {discount}%</span><br>
                <span style="color:#aaa;">Lowering {selected_commodity} to ₹{latest_price * (1 - discount/100):.2f}/kg optimizes clearance velocity within {days_to_spoil} days.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"Inventory condition safe. Maintain standard pricing structure.")

# --- APP ROUTING ---
if st.session_state.logged_in:
    main_dashboard()
else:
    login_page()