import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Natural Gas Price Forecasting", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('daily_csv.csv')
        # Convert date column if needed
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        return data
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        st.warning("Sample data loaded. Upload 'daily_csv.csv' for your actual data.")
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        prices = 2.5 + 0.01 * np.arange(len(dates)) + np.random.normal(0, 0.5, len(dates))
        data = pd.DataFrame({'Date': dates, 'Price': prices})
        return data

data = load_data()

# Title
st.title("🔮 Natural Gas Price Forecasting Dashboard")
st.markdown("---")

# Display data info
st.sidebar.header("📊 Data Overview")
st.sidebar.write(f"**Total Records:** {len(data):,}")
st.sidebar.write(f"**Date Range:** {data['Date'].min().date()} to {data['Date'].max().date()}")
st.sidebar.write(f"**Columns:** {', '.join(data.columns)}")

# Sidebar filters
st.sidebar.header("🔍 Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    [data['Date'].min().date(), data['Date'].max().date()]
)

if len(date_range) == 2:
    filtered_data = data[(data['Date'] >= pd.to_datetime(date_range[0])) & 
                         (data['Date'] <= pd.to_datetime(date_range[1]))]
else:
    filtered_data = data

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Analysis", "🤖 Forecasting", "📋 Raw Data"])

# Tab 1: Time Series
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Time Series Analysis")
        
        # Line chart for time series
        fig = px.line(
            filtered_data,
            x='Date',
            y='Price',
            title='Natural Gas Price Over Time',
            labels={'Price': 'Price ($/MMBtu)', 'Date': 'Date'},
            template='plotly_dark'
        )
        
        # Add moving average
        if len(filtered_data) > 30:
            ma_window = st.slider("Moving Average Window", min_value=7, max_value=90, value=30)
            filtered_data['MA'] = filtered_data['Price'].rolling(window=ma_window).mean()
            
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['MA'],
                mode='lines',
                name=f'{ma_window}-Day MA',
                line=dict(color='orange', width=2)
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Statistics")
        st.metric("Current Price", f"${filtered_data['Price'].iloc[-1]:.2f}")
        st.metric("Average Price", f"${filtered_data['Price'].mean():.2f}")
        st.metric("Price Volatility", f"{filtered_data['Price'].std():.2f}")
        
        # Price change
        if len(filtered_data) > 1:
            price_change = filtered_data['Price'].iloc[-1] - filtered_data['Price'].iloc[0]
            st.metric("Price Change", f"${price_change:.2f}")

# Tab 2: Analysis
with tab2:
    st.header("Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            filtered_data,
            x='Price',
            nbins=30,
            title='Price Distribution',
            labels={'Price': 'Price ($/MMBtu)'},
            template='plotly_dark'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Monthly analysis
        filtered_data['Month'] = filtered_data['Date'].dt.month
        monthly_avg = filtered_data.groupby('Month')['Price'].mean().reset_index()
        
        fig_month = px.bar(
            monthly_avg,
            x='Month',
            y='Price',
            title='Average Price by Month',
            labels={'Price': 'Avg Price ($/MMBtu)', 'Month': 'Month'},
            template='plotly_dark'
        )
        st.plotly_chart(fig_month, use_container_width=True)
    
    with col2:
        # Seasonal decomposition (simplified)
        filtered_data['DayOfYear'] = filtered_data['Date'].dt.dayofyear
        seasonal_data = filtered_data.groupby('DayOfYear')['Price'].mean().reset_index()
        
        fig_seasonal = px.line(
            seasonal_data,
            x='DayOfYear',
            y='Price',
            title='Seasonal Pattern (Average by Day of Year)',
            labels={'Price': 'Avg Price ($/MMBtu)', 'DayOfYear': 'Day of Year'},
            template='plotly_dark'
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Correlation heatmap (if multiple features exist)
        if len(data.columns) > 2:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu_r',
                    text_auto=True
                )
                st.plotly_chart(fig_corr, use_container_width=True)

# Tab 3: Forecasting

with tab3:
    st.header("Price Forecasting")
    
    # Prepare data for forecasting
    forecast_data = filtered_data.copy()
    
    # Check for and handle missing values
    if forecast_data['Price'].isna().any():
        st.warning(f"Found {forecast_data['Price'].isna().sum()} missing values in Price column.")
        
        # Create options for handling missing values
        missing_strategy = st.selectbox(
            "Handle missing values by:",
            ["Forward Fill", "Backward Fill", "Linear Interpolation", "Drop Missing"]
        )
        
        if missing_strategy == "Forward Fill":
            forecast_data['Price'] = forecast_data['Price'].ffill()
        elif missing_strategy == "Backward Fill":
            forecast_data['Price'] = forecast_data['Price'].bfill()
        elif missing_strategy == "Linear Interpolation":
            forecast_data['Price'] = forecast_data['Price'].interpolate(method='linear')
        else:  # Drop Missing
            forecast_data = forecast_data.dropna(subset=['Price'])
        
        st.success(f"Missing values handled using {missing_strategy}")
    
    # Create days feature
    forecast_data['Days'] = (forecast_data['Date'] - forecast_data['Date'].min()).dt.days
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model selection
        model_type = st.selectbox(
            "Select Forecasting Model",
            ["Linear Regression", "Random Forest"]
        )
        
        forecast_days = st.slider("Days to Forecast", min_value=7, max_value=365, value=30)
        
        # Check if we have enough data
        if len(forecast_data) < 20:
            st.error(f"Not enough data for forecasting. Need at least 20 data points, but only have {len(forecast_data)}.")
        else:
            # Prepare features
            X = forecast_data[['Days']].values
            y = forecast_data['Price'].values
            
            # Double-check for NaN values
            if np.isnan(X).any() or np.isnan(y).any():
                st.error("Data still contains NaN values. Please handle missing values first.")
            else:
                if st.button("Train Model & Generate Forecast"):
                    # Train model
                    if model_type == "Linear Regression":
                        model = LinearRegression()
                    else:
                        model = RandomForestRegressor(
                            n_estimators=100, 
                            random_state=42,
                            max_depth=10
                        )
                    
                    # Split data - for time series, we use sequential split
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    r2 = r2_score(y_test, y_pred_test)
                    
                    # Generate forecast
                    last_day = forecast_data['Days'].max()
                    future_days = np.arange(last_day + 1, last_day + forecast_days + 1).reshape(-1, 1)
                    future_prices = model.predict(future_days)
                    
                    # Create future dates
                    last_date = forecast_data['Date'].max()
                    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                    
                    # Create forecast dataframe
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Price': future_prices,
                        'Type': 'Forecast'
                    })
                    
                    # Combine with historical data
                    historical_df = forecast_data[['Date', 'Price']].copy()
                    historical_df['Type'] = 'Historical'
                    
                    combined_df = pd.concat([historical_df, forecast_df])
                    
                    # Plot forecast
                    fig_forecast = px.line(
                        combined_df,
                        x='Date',
                        y='Price',
                        color='Type',
                        title=f'Natural Gas Price Forecast ({forecast_days} days)',
                        labels={'Price': 'Price ($/MMBtu)', 'Date': 'Date'},
                        template='plotly_dark',
                        color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
                    )
                    
                    # Add confidence interval
                    if model_type == "Linear Regression":
                        # For linear regression, we can calculate confidence intervals
                        from scipy import stats
                        
                        # Get predictions for entire dataset
                        all_predictions = model.predict(X)
                        residuals = y - all_predictions
                        residual_std = np.std(residuals)
                        
                        forecast_df['Upper'] = forecast_df['Price'] + 1.96 * residual_std
                        forecast_df['Lower'] = forecast_df['Price'] - 1.96 * residual_std
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                            y=forecast_df['Upper'].tolist() + forecast_df['Lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval'
                        ))
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("MAE", f"${mae:.3f}")
                    with col_metric2:
                        st.metric("RMSE", f"${rmse:.3f}")
                    with col_metric3:
                        st.metric("R² Score", f"{r2:.3f}")
                    
                    # Show forecast values
                    st.subheader("Forecast Values")
                    forecast_display = forecast_df.copy()
                    forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                    forecast_display['Price'] = forecast_display['Price'].apply(lambda x: f"${x:.2f}")
                    st.dataframe(forecast_display[['Date', 'Price']], hide_index=True)
                    
                    # Download forecast
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast",
                        data=csv,
                        file_name="gas_price_forecast.csv",
                        mime="text/csv"
                    )
# Tab 4: Raw Data
with tab4:
    st.header("Raw Data")
    
    # Data display options
    show_cols = st.multiselect(
        "Select columns to display",
        options=data.columns.tolist(),
        default=['Date', 'Price']
    )
    
    # Show data
    if show_cols:
        st.dataframe(filtered_data[show_cols], use_container_width=True)
    
    # Data statistics
    st.subheader("Data Statistics")
    st.dataframe(filtered_data.describe(), use_container_width=True)
    
    # Download data
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_gas_prices.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Natural Gas Price Forecasting Dashboard | Updated: " + datetime.now().strftime("%Y-%m-%d"))