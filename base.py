import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI


client = OpenAI(
  api_key="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"
)


def create_chart(opt):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Load stock data from CSV
    data = pd.read_csv("dash-stock-ticker-demo.csv") 
    prices = data[opt].values
    dates = pd.to_datetime(data['Date'].values, errors='coerce')  # Convert to datetime format

    # Normalize prices for better training stability using robust scaling
    prices_normalized = (prices - np.median(prices)) / np.std(prices)

    price_end = prices[-1]
    price_beg = prices[0]
    positive = 1
    if(price_end <= price_beg):
        positive = 0
    
    # Create features and targets using sliding window
    def create_features(prices, window_size):
        X, y = [], []
        for i in range(len(prices) - window_size):
            X.append(prices[i:i + window_size])  # Features: `window_size` past prices
            y.append(prices[i + window_size])   # Target: The next day's price
        return np.array(X), np.array(y)

    # Define sliding window size
    window_size = 10  # Adjust this value based on data trends
    X, y = create_features(prices_normalized, window_size)

    # Use all data for training (no randomization)
    X_train = X
    y_train = y

    # Reshape X_train to have shape (n_samples, window_size)
    X_train = X_train.reshape(-1, window_size)

    # Make sure y_train is a column vector (n_samples, 1)
    y_train = y_train.reshape(-1, 1)

    # Initialize parameters
    weights = np.random.randn(window_size, 1)  # Shape (window_size, 1)
    bias = np.random.randn(1)                  # Shape (1,)
    learning_rate = 0.01
    epochs = 5000
    
    # Prediction function
    def predict(X, weights, bias):
        return np.dot(X, weights) + bias

    # Loss function
    def compute_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Gradient descent
    def gradient_descent(X, y, weights, bias, learning_rate):
        n = len(y)
        y_pred = predict(X, weights, bias)
        
        # Compute gradients for weights
        dw = -(2 / n) * np.dot(X.T, (y - y_pred))  # Gradient for weights (shape (window_size, 1))
        db = -(2 / n) * np.sum(y - y_pred)         # Gradient for bias (scalar)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        return weights, bias

    # Train the model
    for epoch in range(epochs):
        y_pred = predict(X_train, weights, bias)
        loss = compute_loss(y_train, y_pred)
        weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate)

    # Evaluate on all data (using the trained model to predict on the entire dataset)
    y_train_pred = predict(X_train, weights, bias)
    train_loss = compute_loss(y_train, y_train_pred)

    # Denormalize predictions and actual prices for comparison
    y_actual = y_train * np.std(prices) + np.median(prices)
    y_pred_actual = y_train_pred * np.std(prices) + np.median(prices)

    # Use the corresponding dates for the entire dataset
    all_dates = dates[window_size:]  # Get the correct dates for the full dataset (after the first window_size days)

    # Predict future values for the next 365 days
    y_pred_actual_1d = y_pred_actual.flatten()

    # Calculate the average rate of change from the model's predictions
    pred_changes = np.diff(y_pred_actual_1d) / y_pred_actual_1d[:-1]  # Daily percentage changes
    avg_daily_change = np.mean(pred_changes)
    std_daily_change = np.std(pred_changes)

    n_future_days = 365 + window_size
    future_predictions = np.zeros(n_future_days)
    future_predictions[0] = y_actual[-1, 0]  # Explicitly select the scalar element
    for a in range(0, window_size):
        future_predictions[a] = y_actual[a - 10, 0]  # Ensure correct indexing

    # messages_input = []
    
    # messages_input.append({"role": "system", "content": f"Actual Data: Dates: {dates} - Prices: {prices}"})
    #     #messages_input.append({"role": "system", "content": ("Actual Data: Date: ", dates[x], " - Price: ", prices[x])})
    # # Generate future predictions using the learned pattern plus controlled noise
    # messages_input.append({"role": "user", "content": f"Based off the past historical data and the newer predictions does the next predicted stock price make sense for the following company, {opt}? If it does, give me the predicted stock price; if it doesn't, give me your predicted stock price. DO NOT GIVE ANYTHING OTHER THAN STOCK PRICES AS A RETURN: {102}"})
    # completion = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     store=True,
    #     messages= messages_input,
    # )
    # messages_input.pop
    # print(completion.choices[0].text)
    for i in range(window_size, n_future_days):
        l10 = future_predictions[i-window_size:i]
        base_change = predict(l10.reshape(-1,window_size), weights, bias)[0]
        temp = base_change[0] - future_predictions[i - window_size]
        if (std_daily_change < abs(temp)):
            temp = temp * 0.5
        base_change = future_predictions[i - window_size] + temp
        #messages_input.append({"role": "user", "content": ("Based off the past historical data and the newer predictions does the next predicted stock price make sense for the following company,", opt, " if it does give me the predicted stock price, if it doesn't give me your predicted stock price, DO NOT GIVE ANYTHING OTHER THEN STOCK PRICES AS A RETURN", base_change)})
        # messages_input.append({"role": "user", "content": f"Based off the past historical data and the newer predictions does the next predicted stock price make sense for the following company, {opt}? If it does, give me the predicted stock price; if it doesn't, give me your predicted stock price. DO NOT GIVE ANYTHING OTHER THAN STOCK PRICES AS A RETURN: {base_change}"})
        # completion = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     store=True,
        #     messages= messages_input,
        # )
        # messages_input.pop
        # print(completion.choices[0].text)
        # print(base_change, "    ", np.mean(prices) + np.std(prices)/2)
        if positive and base_change < (np.mean(prices) + np.std(prices)/2):
            base_change = future_predictions[i - 5] + abs(np.random.normal(0, std_daily_change * 0.5))
            future_predictions[i] = base_change
        elif abs(future_predictions[9] - base_change) > std_daily_change:
            base_change = avg_daily_change
            noise = np.random.normal(0, std_daily_change * 0.5)
            daily_return = base_change + noise
            daily_return = np.clip(daily_return, -0.1, 0.1) 
            dampening = np.exp(-i / (n_future_days * 2))
            if daily_return > 0:
                future_predictions[i] = future_predictions[i-1] * (1 + daily_return * dampening)
            else:
                future_predictions[i] = future_predictions[i-1] * (1 + daily_return)
        else:
            future_predictions[i] = base_change


    # Generate future dates
    last_date = dates[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_future_days - window_size, freq='D')

    # Calculate min and max prices for setting y-axis limits
    all_prices = np.concatenate([prices, future_predictions[window_size:]])
    price_min = all_prices.min()
    price_max = all_prices.max()
    leeway = 0.1 * (price_max - price_min)  # 10% leeway

    # Set y-axis limits
    y_min = price_min - leeway
    y_max = price_max + leeway
    
    
    # Plot actual vs predicted prices with future predictions
    ax.plot(all_dates, y_pred_actual_1d, label="Predicted Prices", color="red", alpha=0.7)
    ax.plot(all_dates, y_actual, label="Actual Prices", color="blue")
    ax.plot(future_dates, future_predictions[window_size:], label="Future Predictions", color="green", linestyle='--')
    ax.fill_between(future_dates, future_predictions[window_size:], color='green', alpha=0.2)

    # Add confidence intervals (±2 standard deviations)
    std_range = np.std(y_pred_actual_1d - y_actual.flatten())
    upper_bound = future_predictions[window_size:] + 2 * std_range
    lower_bound = future_predictions[window_size:] - 2 * std_range
    ax.fill_between(future_dates, lower_bound, upper_bound, color='gray', alpha=0.1, label='95% Confidence Interval')

    # Set title and labels
    ax.set_title("Stock Price Prediction with Future Projections")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.set_xticklabels(all_dates, rotation=45)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.6)
    plt.tight_layout()

    return fig, train_loss


create_chart("AAPL")