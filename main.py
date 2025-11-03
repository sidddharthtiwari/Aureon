import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# Page configuration
st.set_page_config(page_title="Aureon - Structure Analysis", layout="wide", page_icon="📊")

# Title and description
st.title("🔷 Aureon - Structure Analysis Platform")
st.markdown("### Comprehensive A-Z Analysis: Seasonality, Risk & Reward Assessment")

# Sidebar for inputs
st.sidebar.header("📋 Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Structure Data (CSV/Excel)", type=['csv', 'xlsx'])

# Input parameters
st.sidebar.subheader("Parameters")
tick_size = st.sidebar.number_input("Tick Size", min_value=0.0001, value=0.01, step=0.0001, format="%.4f")
tick_value = st.sidebar.number_input("Tick Value (Monetary)", min_value=0.01, value=1.0, step=0.01)

# Date range inputs (will be populated after file upload)
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Bracket size for seasonality
bracket_days = st.sidebar.number_input("Bracket Size (Days)", min_value=1, max_value=30, value=7, step=1)

# Helper Functions
def parse_date(date_str):
    """Parse date from string format"""
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except:
        try:
            return pd.to_datetime(date_str)
        except:
            return pd.NaT

def load_and_process_data(file):
    """Load and process the structure data"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def forward_fill_other_years(df):
    """Forward fill all year columns except the first year (recent year - columns 0,1)"""
    if df.shape[1] <= 2:
        return df
    
    # Forward fill all settlement columns starting from column 3 (skip first year)
    for i in range(3, df.shape[1], 2):  # Skip timestamp columns
        if i < df.shape[1]:
            df.iloc[:, i] = df.iloc[:, i].fillna(method='ffill')
    
    return df

def extract_historical_data_with_dates(df, start_idx, end_idx):
    """Extract settlement data for all HISTORICAL years (exclude first year) within date range"""
    year_data = {}
    
    # Start from column 2 (skip first year which is columns 0,1)
    for i in range(2, df.shape[1], 2):
        if i + 1 < df.shape[1]:
            year_col = df.columns[i + 1]
            timestamp_col = df.columns[i]
            
            # Extract settlement values and dates for the date range
            settlements = df.iloc[start_idx:end_idx+1, i+1].values
            dates = df.iloc[start_idx:end_idx+1, i].values
            
            # Get start and end dates for this year
            valid_indices = [idx for idx, val in enumerate(settlements) if pd.notna(val)]
            if valid_indices:
                actual_start_date = dates[valid_indices[0]]
                actual_end_date = dates[valid_indices[-1]]
            else:
                actual_start_date = dates[0] if len(dates) > 0 else None
                actual_end_date = dates[-1] if len(dates) > 0 else None
            
            # Only include if there's any valid data
            if not all(pd.isna(settlements)):
                year_data[year_col] = {
                    'settlements': settlements,
                    'dates': dates,
                    'start_date': actual_start_date,
                    'end_date': actual_end_date
                }
    
    return year_data

def calculate_linear_regression_slope(values):
    """Calculate linear regression slope for trend detection using Cov(t, Settle)/Var(t)"""
    # Remove NA values
    clean_data = [(i, v) for i, v in enumerate(values) if pd.notna(v)]
    
    if len(clean_data) < 2:
        return 0, 0
    
    indices, prices = zip(*clean_data)
    indices = np.array(indices, dtype=float)
    prices = np.array(prices, dtype=float)
    
    # Calculate slope using Cov(t, Settle) / Var(t)
    mean_t = np.mean(indices)
    mean_p = np.mean(prices)
    
    cov = np.sum((indices - mean_t) * (prices - mean_p)) / len(indices)
    var = np.sum((indices - mean_t) ** 2) / len(indices)
    
    slope = cov / var if var != 0 else 0
    
    # Calculate strength (normalized end-to-end change)
    std = np.std(prices) if len(prices) > 1 else 1
    strength = (prices[-1] - prices[0]) / std if std != 0 else 0
    
    return slope, strength

def calculate_dynamic_threshold(values):
    """Calculate dynamic threshold as 0.1 × rolling_std"""
    clean_values = [v for v in values if pd.notna(v)]
    if len(clean_values) > 1:
        rolling_std = np.std(clean_values)
        return 0.1 * rolling_std
    return 0.01

def classify_trend_regression(slope, strength, threshold):
    """Classify trend using regression method"""
    if slope > threshold and strength > 0.5:
        return "Bullish"
    elif slope < -threshold and strength < -0.5:
        return "Bearish"
    else:
        return "Sideways"

def classify_trend_basic(values):
    """Classify trend using basic method: end_price - start_price"""
    clean_values = [v for v in values if pd.notna(v)]
    if len(clean_values) < 2:
        return "Sideways"
    
    start_price = clean_values[0]
    end_price = clean_values[-1]
    diff = end_price - start_price
    
    # Use a small threshold for "near 0"
    threshold = 0.0001
    
    if diff > threshold:
        return "Bullish"
    elif diff < -threshold:
        return "Bearish"
    else:
        return "Sideways"

def calculate_risk_reward_detailed(settlements, direction, tick_size):
    """Calculate detailed risk and reward using ABS(ABS(start) - ABS(adverse_price)) / tick_size"""
    valid_prices = [p for p in settlements if pd.notna(p)]
    
    if len(valid_prices) < 2:
        return None, None, None, None
    
    start_price = valid_prices[0]
    
    if direction == "Bullish":
        # Risk: maximum adverse movement (lowest point)
        min_price = min(valid_prices)
        risk_ticks = abs(abs(start_price) - abs(min_price)) / tick_size
        risk_amount = abs(start_price - min_price)
        
        # Reward: maximum favorable movement (highest point)
        max_price = max(valid_prices)
        reward_ticks = abs(abs(start_price) - abs(max_price)) / tick_size
        reward_amount = abs(max_price - start_price)
        
        return risk_ticks, risk_amount, reward_ticks, reward_amount
        
    elif direction == "Bearish":
        # Risk: maximum adverse movement (highest point)
        max_price = max(valid_prices)
        risk_ticks = abs(abs(start_price) - abs(max_price)) / tick_size
        risk_amount = abs(max_price - start_price)
        
        # Reward: maximum favorable movement (lowest point)
        min_price = min(valid_prices)
        reward_ticks = abs(abs(start_price) - abs(min_price)) / tick_size
        reward_amount = abs(start_price - min_price)
        
        return risk_ticks, risk_amount, reward_ticks, reward_amount
    
    else:  # Sideways
        max_price = max(valid_prices)
        min_price = min(valid_prices)
        risk_amount = max(abs(max_price - start_price), abs(start_price - min_price))
        risk_ticks = abs(abs(start_price) - abs(max_price if abs(max_price - start_price) > abs(min_price - start_price) else min_price)) / tick_size
        return risk_ticks, risk_amount, None, None

def calculate_atr(settlements):
    """Calculate ATR using only settlement prices (absolute differences)"""
    valid_prices = [p for p in settlements if pd.notna(p)]
    
    if len(valid_prices) < 2:
        return None
    
    tr_values = []
    for i in range(1, len(valid_prices)):
        tr = abs(valid_prices[i] - valid_prices[i-1])
        tr_values.append(tr)
    
    atr = np.mean(tr_values) if tr_values else None
    return atr

def analyze_brackets(year_data, bracket_size):
    """Analyze all brackets and return detailed information"""
    # Get the maximum length across all years
    max_len = max(len(data['settlements']) for data in year_data.values())
    
    all_brackets = []
    
    # Create brackets
    bracket_num = 1
    for start in range(0, max_len, bracket_size):
        end = min(start + bracket_size, max_len)
        
        bracket_info = {
            'bracket_num': bracket_num,
            'start_idx': start,
            'end_idx': end,
            'regression_trends': {},
            'basic_trends': {}
        }
        
        for year, year_info in year_data.items():
            settlements = year_info['settlements']
            if end <= len(settlements):
                bracket_data = settlements[start:end]
                
                # Regression method
                threshold = calculate_dynamic_threshold(bracket_data)
                slope, strength = calculate_linear_regression_slope(bracket_data)
                reg_trend = classify_trend_regression(slope, strength, threshold)
                
                # Basic method
                basic_trend = classify_trend_basic(bracket_data)
                
                bracket_info['regression_trends'][year] = reg_trend
                bracket_info['basic_trends'][year] = basic_trend
        
        all_brackets.append(bracket_info)
        bracket_num += 1
    
    return all_brackets

def determine_year_seasonality(all_brackets, year):
    """Determine seasonality for a specific year based on bracket analysis"""
    reg_counts = {'Bullish': 0, 'Bearish': 0, 'Sideways': 0}
    basic_counts = {'Bullish': 0, 'Bearish': 0, 'Sideways': 0}
    
    for bracket in all_brackets:
        if year in bracket['regression_trends']:
            reg_counts[bracket['regression_trends'][year]] += 1
        if year in bracket['basic_trends']:
            basic_counts[bracket['basic_trends'][year]] += 1
    
    # Determine dominant trend for regression method
    reg_total = sum(reg_counts.values())
    if reg_total > 0:
        reg_dominant = max(reg_counts, key=reg_counts.get)
        reg_probability = (reg_counts[reg_dominant] / reg_total * 100)
    else:
        reg_dominant = "Unknown"
        reg_probability = 0
    
    # Determine dominant trend for basic method
    basic_total = sum(basic_counts.values())
    if basic_total > 0:
        basic_dominant = max(basic_counts, key=basic_counts.get)
        basic_probability = (basic_counts[basic_dominant] / basic_total * 100)
    else:
        basic_dominant = "Unknown"
        basic_probability = 0
    
    return {
        'regression': {'trend': reg_dominant, 'probability': reg_probability, 'counts': reg_counts},
        'basic': {'trend': basic_dominant, 'probability': basic_probability, 'counts': basic_counts}
    }

# Main Application Logic
if uploaded_file is not None:
    # Load data
    df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Show raw data preview
        with st.expander("📊 View Raw Data Preview"):
            st.dataframe(df.head(20))
        
        # Parse dates in first column
        df.iloc[:, 0] = df.iloc[:, 0].apply(parse_date)
        
        # Forward fill data (skip first year)
        with st.spinner("Processing data..."):
            df = forward_fill_other_years(df)
        
        st.success("✅ Data preprocessing completed!")
        
        # Show processed data
        with st.expander("📈 View Processed Data Preview"):
            st.dataframe(df.head(20))
        
        # Analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            try:
                # Find date indices
                start_idx = df[df.iloc[:, 0] == pd.to_datetime(start_date)].index
                end_idx = df[df.iloc[:, 0] == pd.to_datetime(end_date)].index
                
                if len(start_idx) == 0 or len(end_idx) == 0:
                    st.error("❌ Start or End date not found in the data!")
                else:
                    start_idx = start_idx[0]
                    end_idx = end_idx[0]
                    
                    if start_idx >= end_idx:
                        st.error("❌ Start date must be before end date!")
                    else:
                        st.success(f"✅ Analyzing data from index {start_idx} to {end_idx} ({end_idx - start_idx + 1} days)")
                        
                        # Extract historical year data (exclude first year)
                        year_data = extract_historical_data_with_dates(df, start_idx, end_idx)
                        st.info(f"📊 Found historical data for {len(year_data)} years (excluding recent year)")
                        
                        # Show date ranges for each year
                        st.subheader("📅 Date Ranges for Each Historical Year")
                        date_range_data = []
                        for year, info in year_data.items():
                            date_range_data.append({
                                'Year': year,
                                'Start Date': info['start_date'],
                                'End Date': info['end_date'],
                                'Data Points': sum(1 for x in info['settlements'] if pd.notna(x))
                            })
                        date_range_df = pd.DataFrame(date_range_data)
                        st.dataframe(date_range_df, use_container_width=True)
                        
                        # STEP 1: Seasonality Analysis
                        st.header("📈 STEP 1: Seasonality Analysis")
                        
                        # Analyze all brackets
                        all_brackets = analyze_brackets(year_data, bracket_days)
                        
                        st.subheader("🔍 Bracket-wise Analysis (All Years)")
                        
                        # Show brackets for confirmation
                        for bracket in all_brackets:
                            with st.expander(f"Bracket {bracket['bracket_num']} (Indices {bracket['start_idx']}-{bracket['end_idx']})"):
                                st.markdown("**Regression Method:**")
                                reg_df = pd.DataFrame([
                                    {'Year': year, 'Trend': trend}
                                    for year, trend in bracket['regression_trends'].items()
                                ])
                                st.dataframe(reg_df, use_container_width=True)
                                
                                st.markdown("**Basic Method (End - Start):**")
                                basic_df = pd.DataFrame([
                                    {'Year': year, 'Trend': trend}
                                    for year, trend in bracket['basic_trends'].items()
                                ])
                                st.dataframe(basic_df, use_container_width=True)
                        
                        # Determine seasonality for each year
                        st.subheader("📊 Year-wise Seasonality Summary")
                        
                        year_seasonality = {}
                        seasonality_summary = []
                        
                        for year in year_data.keys():
                            seasonality = determine_year_seasonality(all_brackets, year)
                            year_seasonality[year] = seasonality
                            
                            seasonality_summary.append({
                                'Year': year,
                                'Regression Trend': seasonality['regression']['trend'],
                                'Regression Prob': f"{seasonality['regression']['probability']:.1f}%",
                                'Bullish (R)': seasonality['regression']['counts']['Bullish'],
                                'Bearish (R)': seasonality['regression']['counts']['Bearish'],
                                'Sideways (R)': seasonality['regression']['counts']['Sideways'],
                                'Basic Trend': seasonality['basic']['trend'],
                                'Basic Prob': f"{seasonality['basic']['probability']:.1f}%",
                                'Bullish (B)': seasonality['basic']['counts']['Bullish'],
                                'Bearish (B)': seasonality['basic']['counts']['Bearish'],
                                'Sideways (B)': seasonality['basic']['counts']['Sideways']
                            })
                        
                        seasonality_df = pd.DataFrame(seasonality_summary)
                        st.dataframe(seasonality_df, use_container_width=True)
                        
                        # Determine OVERALL WINNING DIRECTION
                        st.subheader("🎯 Overall Seasonality Direction")
                        
                        # Count trends across all years for both methods
                        reg_overall = {'Bullish': 0, 'Bearish': 0, 'Sideways': 0}
                        basic_overall = {'Bullish': 0, 'Bearish': 0, 'Sideways': 0}
                        
                        for year, seasonality in year_seasonality.items():
                            reg_overall[seasonality['regression']['trend']] += 1
                            basic_overall[seasonality['basic']['trend']] += 1
                        
                        # Determine winning direction for each method
                        reg_winner = max(reg_overall, key=reg_overall.get)
                        reg_winner_prob = (reg_overall[reg_winner] / sum(reg_overall.values()) * 100) if sum(reg_overall.values()) > 0 else 0
                        
                        basic_winner = max(basic_overall, key=basic_overall.get)
                        basic_winner_prob = (basic_overall[basic_winner] / sum(basic_overall.values()) * 100) if sum(basic_overall.values()) > 0 else 0
                        
                        # Decide final winning direction (regression gets priority)
                        if reg_winner == basic_winner:
                            final_winner = reg_winner
                            confidence = "High (Both methods agree)"
                        else:
                            final_winner = reg_winner
                            confidence = f"Medium (Regression: {reg_winner}, Basic: {basic_winner})"
                        
                        # Display winning direction
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### 📊 Regression Method")
                            st.markdown(f"**Direction:** `{reg_winner}`")
                            st.markdown(f"**Probability:** `{reg_winner_prob:.1f}%`")
                            st.markdown(f"**Years:** Bullish: {reg_overall['Bullish']}, Bearish: {reg_overall['Bearish']}, Sideways: {reg_overall['Sideways']}")
                        
                        with col2:
                            st.markdown("### 📊 Basic Method")
                            st.markdown(f"**Direction:** `{basic_winner}`")
                            st.markdown(f"**Probability:** `{basic_winner_prob:.1f}%`")
                            st.markdown(f"**Years:** Bullish: {basic_overall['Bullish']}, Bearish: {basic_overall['Bearish']}, Sideways: {basic_overall['Sideways']}")
                        
                        with col3:
                            st.markdown("### 🏆 WINNING DIRECTION")
                            direction_emoji = "📈" if final_winner == "Bullish" else "📉" if final_winner == "Bearish" else "↔️"
                            st.markdown(f"# {direction_emoji} **{final_winner}**")
                            st.markdown(f"**Confidence:** {confidence}")
                        
                        st.markdown("---")
                        
                        # Risk & Reward Analysis for ALL YEARS
                        st.header(f"⚠️ STEP 2: Risk & Reward Analysis - All Years")
                        
                        st.info(f"Analyzing all historical years. Winning direction: **{final_winner}**")
                        
                        # Separate years into favorable and unfavorable
                        favorable_years_data = {}
                        unfavorable_years_data = {}
                        
                        for year, info in year_data.items():
                            year_trend = year_seasonality[year]['regression']['trend']
                            if year_trend == final_winner:
                                favorable_years_data[year] = info
                            else:
                                unfavorable_years_data[year] = info
                        
                        st.success(f"Found {len(favorable_years_data)} favorable years ({final_winner}) and {len(unfavorable_years_data)} unfavorable years")
                        
                        # Calculate risk & reward for FAVORABLE years
                        favorable_year_results = []
                        
                        for year, info in favorable_years_data.items():
                            settlements = info['settlements']
                            
                            risk_ticks, risk_amount, reward_ticks, reward_amount = calculate_risk_reward_detailed(
                                settlements, 
                                final_winner, 
                                tick_size
                            )
                            atr = calculate_atr(settlements)
                            
                            # Get start, end and extreme prices for display
                            valid_prices = [p for p in settlements if pd.notna(p)]
                            start_price = valid_prices[0] if valid_prices else None
                            end_price = valid_prices[-1] if valid_prices else None
                            min_price = min(valid_prices) if valid_prices else None
                            max_price = max(valid_prices) if valid_prices else None
                            
                            favorable_year_results.append({
                                'Year': year,
                                'Type': 'Favorable',
                                'Start Price': start_price,
                                'End Price': end_price,
                                'Min Price': min_price,
                                'Max Price': max_price,
                                'Risk (Ticks)': risk_ticks,
                                'Risk (Amount)': risk_amount,
                                'Risk ($)': risk_ticks * tick_value if risk_ticks is not None else None,
                                'Reward (Ticks)': reward_ticks,
                                'Reward (Amount)': reward_amount,
                                'Reward ($)': reward_ticks * tick_value if reward_ticks is not None else None,
                                'ATR': atr,
                                'Risk:Reward': f"1:{reward_ticks/risk_ticks:.2f}" if risk_ticks and reward_ticks and risk_ticks > 0 else 'N/A'
                            })
                        
                        # Calculate ONLY RISK for UNFAVORABLE years
                        unfavorable_year_results = []
                        
                        for year, info in unfavorable_years_data.items():
                            settlements = info['settlements']
                            
                            risk_ticks, risk_amount, _, _ = calculate_risk_reward_detailed(
                                settlements, 
                                final_winner,  # Use winning direction to calculate risk
                                tick_size
                            )
                            atr = calculate_atr(settlements)
                            
                            # Get start, end and extreme prices for display
                            valid_prices = [p for p in settlements if pd.notna(p)]
                            start_price = valid_prices[0] if valid_prices else None
                            end_price = valid_prices[-1] if valid_prices else None
                            min_price = min(valid_prices) if valid_prices else None
                            max_price = max(valid_prices) if valid_prices else None
                            
                            unfavorable_year_results.append({
                                'Year': year,
                                'Type': 'Unfavorable',
                                'Start Price': start_price,
                                'End Price': end_price,
                                'Min Price': min_price,
                                'Max Price': max_price,
                                'Risk (Ticks)': risk_ticks,
                                'Risk (Amount)': risk_amount,
                                'Risk ($)': risk_ticks * tick_value if risk_ticks is not None else None,
                                'Reward (Ticks)': None,
                                'Reward (Amount)': None,
                                'Reward ($)': None,
                                'ATR': atr,
                                'Risk:Reward': 'N/A'
                            })
                        
                        # Combine all results
                        all_year_results = favorable_year_results + unfavorable_year_results
                        
                        # Display all years data
                        st.subheader(f"📋 All Years - Detailed Analysis")
                        display_df = pd.DataFrame([
                            {
                                'Year': r['Year'],
                                'Type': r['Type'],
                                'Start Price': f"{r['Start Price']:.4f}" if r['Start Price'] is not None else 'N/A',
                                'End Price': f"{r['End Price']:.4f}" if r['End Price'] is not None else 'N/A',
                                'Min Price': f"{r['Min Price']:.4f}" if r['Min Price'] is not None else 'N/A',
                                'Max Price': f"{r['Max Price']:.4f}" if r['Max Price'] is not None else 'N/A',
                                'Risk (Ticks)': f"{r['Risk (Ticks)']:.2f}" if r['Risk (Ticks)'] is not None else 'N/A',
                                'Risk ($)': f"${r['Risk ($)']:.2f}" if r['Risk ($)'] is not None else 'N/A',
                                'Reward (Ticks)': f"{r['Reward (Ticks)']:.2f}" if r['Reward (Ticks)'] is not None else 'N/A',
                                'Reward ($)': f"${r['Reward ($)']:.2f}" if r['Reward ($)'] is not None else 'N/A',
                                'Risk:Reward': r['Risk:Reward'],
                                'ATR': f"{r['ATR']:.4f}" if r['ATR'] is not None else 'N/A'
                            }
                            for r in all_year_results
                        ])
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Statistics for favorable years only
                        st.subheader(f"📊 Favorable Years ({final_winner}) - Statistical Summary")
                        
                        risk_ticks_list = [r['Risk (Ticks)'] for r in favorable_year_results if r['Risk (Ticks)'] is not None]
                        reward_ticks_list = [r['Reward (Ticks)'] for r in favorable_year_results if r['Reward (Ticks)'] is not None]
                        rr_ratios = [r['Reward (Ticks)']/r['Risk (Ticks)'] for r in favorable_year_results if r['Risk (Ticks)'] is not None and r['Reward (Ticks)'] is not None and r['Risk (Ticks)'] > 0]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### 🔴 Risk Statistics")
                            if risk_ticks_list:
                                st.metric("Max Risk (Ticks)", f"{np.max(risk_ticks_list):.2f}")
                                st.metric("Max Risk ($)", f"${np.max(risk_ticks_list) * tick_value:.2f}")
                                st.metric("Mean Risk (Ticks)", f"{np.mean(risk_ticks_list):.2f}")
                                st.metric("Mean Risk ($)", f"${np.mean(risk_ticks_list) * tick_value:.2f}")
                                st.metric("Median Risk (Ticks)", f"{np.median(risk_ticks_list):.2f}")
                                st.metric("Median Risk ($)", f"${np.median(risk_ticks_list) * tick_value:.2f}")
                            else:
                                st.warning("No risk data available")
                        
                        with col2:
                            st.markdown("### 🟢 Reward Statistics")
                            if reward_ticks_list:
                                st.metric("Max Reward (Ticks)", f"{np.max(reward_ticks_list):.2f}")
                                st.metric("Max Reward ($)", f"${np.max(reward_ticks_list) * tick_value:.2f}")
                                st.metric("Mean Reward (Ticks)", f"{np.mean(reward_ticks_list):.2f}")
                                st.metric("Mean Reward ($)", f"${np.mean(reward_ticks_list) * tick_value:.2f}")
                                st.metric("Median Reward (Ticks)", f"{np.median(reward_ticks_list):.2f}")
                                st.metric("Median Reward ($)", f"${np.median(reward_ticks_list) * tick_value:.2f}")
                            else:
                                st.warning("No reward data available")
                        
                        with col3:
                            st.markdown("### ⚖️ Risk:Reward Ratios")
                            if rr_ratios:
                                st.metric("Mean R:R", f"1:{np.mean(rr_ratios):.2f}")
                                st.metric("Median R:R", f"1:{np.median(rr_ratios):.2f}")
                                st.metric("Best R:R", f"1:{np.max(rr_ratios):.2f}")
                                st.metric("Worst R:R", f"1:{np.min(rr_ratios):.2f}")
                            else:
                                st.warning("No R:R data available")
                        
                        # ATR Analysis for all years
                        st.subheader(f"📊 ATR Analysis - All Years")
                        
                        atr_table = []
                        for result in all_year_results:
                            atr_table.append({
                                'Year': result['Year'],
                                'Type': result['Type'],
                                'ATR': f"{result['ATR']:.4f}" if result['ATR'] is not None else 'N/A',
                                'Risk (Ticks)': f"{result['Risk (Ticks)']:.2f}" if result['Risk (Ticks)'] is not None else 'N/A'
                            })
                        
                        atr_df = pd.DataFrame(atr_table)
                        st.dataframe(atr_df, use_container_width=True)
                        
                        # Average ATR for favorable and unfavorable years
                        favorable_atrs = [r['ATR'] for r in favorable_year_results if r['ATR'] is not None]
                        unfavorable_atrs = [r['ATR'] for r in unfavorable_year_results if r['ATR'] is not None]
                        
                        if favorable_atrs or unfavorable_atrs:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                if favorable_atrs:
                                    st.metric(f"📈 Avg ATR (Favorable)", f"{np.mean(favorable_atrs):.4f}")
                            with col2:
                                if favorable_atrs:
                                    st.metric(f"📈 Median ATR (Favorable)", f"{np.median(favorable_atrs):.4f}")
                            with col3:
                                if unfavorable_atrs:
                                    st.metric(f"📉 Avg ATR (Unfavorable)", f"{np.mean(unfavorable_atrs):.4f}")
                            with col4:
                                if unfavorable_atrs:
                                    st.metric(f"📉 Median ATR (Unfavorable)", f"{np.median(unfavorable_atrs):.4f}")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

else:
    st.info("👈 Please upload a CSV or Excel file to begin analysis")
    
    # Show sample format
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your file should have columns in pairs:
    - **Timestamp.YYYY** (Date column)
    - **YYYY** (Settlement price column)
    
    Example:
    ```
    Timestamp.2026 | 2026 | Timestamp.2025 | 2025 | ...
    6/16/2024      | NA   | 6/16/2023      | 0    | ...
    6/17/2024      | 0.01 | 6/17/2023      | NA   | ...
    ```
    
     **Note:** The first year (columns 0,1) will be excluded from analysis.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About Aureon")
st.sidebar.info("""
**Aureon** - Comprehensive structure analysis tool for:
- Seasonality detection (Regression + Basic methods)
- Risk & reward assessment in ticks and dollars
- ATR-based volatility analysis
- Historical pattern recognition
""")
