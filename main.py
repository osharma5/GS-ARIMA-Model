import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

gs = yf.Ticker('GS')
data = gs.history(period="1y") 

#data cleaning (keep columns as neeeded and remove NAN valyes!)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.dropna(inplace=True) #removing missing values


data.to_csv('gs_historical_data.csv')

#I need only closing prices for the GS Ticker
series = data['Close']
d = 1  # First difference for stationarity

#Creating a function to fit the best ARIMA model with different parameters
def best_arima(ts):
    best_aic = float('inf')
    best_order = None
    best_model = None
    #Used range(4) to do a fast and sufficient computation: 0 to 3 preferred
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                #With different combinations of p and q, try to fit a model which gives the least AIC value (best fit)
                model = ARIMA(ts, order=(p, d, q))
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
                    best_model = fit
            except Exception:
                continue
    return best_order, best_model

order, model = best_arima(series)
print('Best ARIMA order:', order)

# Forecast for the next 30 days: ARIMA model is best for short term forecasting
forecast_steps = 30
future_pred = model.get_forecast(steps=forecast_steps)
forecast = future_pred.predicted_mean
conf_int = future_pred.conf_int()

#Generating a sequence of dates (Business days only) from the last date of our data
future_dates = pd.date_range(series.index[-1], periods=forecast_steps + 1, freq='B')[1:] #removing the first element of the series because that's the last day of current data; we need future dates!
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': forecast.values,
    #SYNTAX REFRESHER -> iloc: locating the elements in dataframe by element position: applicable for pandas dataframe
    'Lower_CI': conf_int.iloc[:, 0].values,
    'Upper_CI': conf_int.iloc[:, 1].values
})
forecast_df.to_csv('gs_forecast_data.csv', index=False)

metrics = {
    "AIC": model.aic,
    "BIC": model.bic,
    #finding mean absolute percentage error (starting the series by 1 because ARIMA loses one oberseravtion due to difference of one)
    "MAPE": np.mean(np.abs(model.resid / series[1:])) * 100,
    "Order": str(order)
}
pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_csv('gs_model_metrics.csv', index=False)

print("Analysis complete! CSV files updated.")