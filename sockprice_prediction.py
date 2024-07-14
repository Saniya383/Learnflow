import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data_path = 'path_to_stock_data.csv'
df = pd.read_csv(data_path)
print(df.head())
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Daily_Return'] = df['Close'].pct_change()
df.dropna(inplace=True)
X = df[['Daily_Return']].values
y = df['Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_rmse = mean_squared_error(y_train, train_pred, squared=False)
test_rmse = mean_squared_error(y_test, test_pred, squared=False)
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual')
plt.plot(df.iloc[X_train.shape[0]:].index, test_pred, label='Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
def predict_next_day_price(model, last_day_return):
    return model.predict([[last_day_return]])
last_day_return = df['Daily_Return'].iloc[-1]
predicted_price = predict_next_day_price(model, last_day_return)
print(f'Predicted price for the next day: ${predicted_price[0]:.2f}')
