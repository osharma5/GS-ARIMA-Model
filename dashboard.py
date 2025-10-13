import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Goldman Sachs ARIMA Dashboard",
    layout="wide"
)

st.title("Goldman Sachs Stock ARIMA Forecasting Dashboard")

historical = pd.read_csv('gs_historical_data.csv', parse_dates=['Date'])
forecast = pd.read_csv('gs_forecast_data.csv', parse_dates=['Date'])
metrics = pd.read_csv('gs_model_metrics.csv')

# Kep performance indicator 
last_close = historical['Close'].iloc[-1]
forecast_price = forecast['Forecast'].iloc[-1]
aic = metrics.loc[metrics['Metric']=='AIC', 'Value'].iloc[0]
mape = metrics.loc[metrics['Metric']=='MAPE', 'Value'].iloc[0]
order = metrics.loc[metrics['Metric']=='Order', 'Value'].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Last Close Price", f"${last_close:,.2f}")
col2.metric("30d Forecast", f"${forecast_price:,.2f}")
col3.metric("Best ARIMA Order", order)

st.write(f"**Model AIC**: {float(aic):.2f}  &nbsp;&nbsp; **MAPE**: {float(mape):.2f}")


# Plot for historic data and also future
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(historical['Date'], historical['Close'], label='Historical Close')
ax.plot(forecast['Date'], forecast['Forecast'], color='green', label='Forecast (30d)')
ax.fill_between(forecast['Date'], forecast['Lower_CI'], forecast['Upper_CI'],
                color='green', alpha=0.2, label='Confidence Interval')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.subheader("Forecast Table (next 30 trading days)")
st.dataframe(forecast)


