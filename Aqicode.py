import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('aqi_dataset.csv')

print(df.head())
print(df.info())
print(df.columns)

df["Date"] = pd.to_datetime(df["Date"])

#DATA CLEANING

print("Null values in each column:")
print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

df = df.dropna()

print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df.shape}")

data_filled = df.fillna(0)
print("Null values after cleaning:")
print(df.isnull().sum())


#feature engineering
highest_aqi_date = df.groupby('Date')['AQI'].mean().idxmax()
highest_aqi_value = df.groupby('Date')['AQI'].mean().max()

print(f"The date with the highest average AQI is {highest_aqi_date} with an AQI of {highest_aqi_value:.2f}")

city_highest_pm25 = df.groupby('City')['PM2.5'].mean().idxmax()
highest_pm25_value = df.groupby('City')['PM2.5'].mean().max()

print(f"The city with the highest average PM2.5 is {city_highest_pm25} with PM2.5 concentration of {highest_pm25_value:.2f}")

pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
pollutants_avg = df[pollutants].mean()

print("Overall Average Values for Key Pollutants:")
print(pollutants_avg)


severe_aqi_days = (df['AQI'] > 300).sum()
print(f"The number of days with AQI greater than 300 (severe) is {severe_aqi_days}.")


city_aqi_variance = df.groupby('City')['AQI'].var().sort_values(ascending=False).head(5)

print("Top 5 Cities with Maximum AQI Variance:")
print(city_aqi_variance)

#EDA

plt.figure(figsize=(10,6))
sns.histplot(df['AQI'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Air Quality Index (AQI')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.show()


city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
city_aqi.plot(kind='bar',color='coral')
plt.title('Top 10 Cities with the highest average AQI')
plt.xlabel('City')
plt.ylabel('Average AQI')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,6))
df.groupby('Date')['AQI'].mean().plot(color='red')
plt.title('AQI Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Average AQI')
plt.show()


plt.figure(figsize=(12,6))
sns.kdeplot(df['PM2.5'], label='PM2.5', color='blue', fill=True)
sns.kdeplot(df['PM10'], label='PM10', color='orange', fill=True)

plt.title('Comparison of PM2.5 and PM10 Levels')
plt.xlabel('Concentration')
plt.ylabel('Density')
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='AQI_Bucket', data=df, palette='viridis') 
plt.title('Distribution of AQI Buckets')
plt.xlabel('AQI Bucket')
plt.ylabel('Count')
plt.show()


cor_map = df[['PM2.5','PM10','AQI']]
plt.figure(figsize=(10,8))
sns.heatmap(cor_map.corr(),annot=True,cmap='coolwarm')
plt.title('correlation heatmap')
plt.show()



# ARIMA Forecast (next 30 days)
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
aqi_series = df['AQI']

from statsmodels.tsa.arima.model import ARIMA

aqi_last100 = aqi_series[-100:]
model = ARIMA(aqi_last100, order=(2,1,2))
model_fit = model.fit()
forecast_arima = model_fit.forecast(steps=30)

last_date = aqi_last100.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

plt.figure(figsize=(12,6))
plt.plot(aqi_last100.index, aqi_last100, label='Actual AQI (Last 100 Days)')
plt.plot(future_dates, forecast_arima, label='ARIMA Forecast (Next 30 Days)', linestyle='--')
plt.title('AQI Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()

#PyTorch LSTM forecast
aqi_values = aqi_series.values.reshape(-1, 1)
scaler = MinMaxScaler()
aqi_scaled = scaler.fit_transform(aqi_values)

look_back = 10
train_lstm = aqi_scaled[-100:]

def create_sequences_torch(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back, 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences_torch(train_lstm, look_back)
X_lstm = torch.tensor(X_lstm, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, look_back, 1)
y_lstm = torch.tensor(y_lstm, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, 1)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_lstm)
    loss = criterion(output, y_lstm)
    loss.backward()
    optimizer.step()


model.eval()
forecast_lstm = []
input_seq = train_lstm[-look_back:].reshape(1, look_back, 1)
input_seq = torch.tensor(input_seq, dtype=torch.float32)
for _ in range(30):
    with torch.no_grad():
        next_val = model(input_seq).item()
    forecast_lstm.append(next_val)
    next_input = torch.tensor([[[next_val]]], dtype=torch.float32)
    input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)

forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm).reshape(-1,1)).flatten()

plt.figure(figsize=(12,6))
plt.plot(aqi_last100.index, scaler.inverse_transform(train_lstm).flatten(), label='Actual AQI (Last 100 Days)')
plt.plot(future_dates, forecast_lstm, label='PyTorch LSTM Forecast (Next 30 Days)', linestyle='--')
plt.title('AQI Forecast using PyTorch LSTM')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()


df.reset_index(inplace=True)

print(df.columns)
SAVE_DF = df.to_csv('cleaned_air_quality_dataset.csv',index=True)