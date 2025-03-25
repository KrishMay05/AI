import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

# Function to create and train the model, then predict future stock prices
def create_chart(option):
    # Load stock data
    data = pd.read_csv("dash-stock-ticker-demo.csv")
    dates = pd.to_datetime(data['Date'].values, errors='coerce')
    prices = data["AAPL"].values.reshape(-1, 1)

    # Normalize prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # Create time-series dataset
    def create_features(prices, window_size):
        X, y = [], []
        for i in range(len(prices) - window_size):
            X.append(prices[i:i + window_size])  # Windowed input
            y.append(prices[i + window_size])    # Next day's price
        return np.array(X), np.array(y)

    window_size = 30
    X, y = create_features(prices_scaled, window_size)

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build Neural Network with more units and layers
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.3),  # Increased dropout rate
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(100, return_sequences=False),  # Additional LSTM layer
        Dropout(0.3),
        Dense(50),  # Increased Dense layer size
        Dropout(0.3),  # Dropout after dense layers
        Dense(1)  # Output layer
    ])

    # Compile model with adjusted learning rate
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())

    # Train model with more epochs
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    # Predict future values
    def predict_future(n_days):
        last_window = X_test[-1].reshape(1, window_size, 1)
        predictions = []
        for _ in range(n_days):
            next_price = model.predict(last_window)[0][0]
            predictions.append(next_price)
            last_window = np.roll(last_window, -1)
            last_window[0, -1, 0] = next_price
        return np.array(predictions)

    # Predict next 365 days
    future_preds_scaled = predict_future(365)
    future_preds = scaler.inverse_transform(future_preds_scaled.reshape(-1, 1))

    # Generate future dates
    start_date = pd.to_datetime("2016-03-01") + pd.Timedelta(days=1)
    future_dates = pd.date_range(start_date, periods=365, freq='D')

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label="Actual Prices", color="blue")
    plt.plot(future_dates, future_preds, label="Future Predictions", color="green", linestyle="--")
    plt.title(f"Stock Price Prediction (LSTM Model) for {option}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # Return the figure object to be used in Streamlit
    return plt.gcf()  # Return the figure for use with Streamlit
