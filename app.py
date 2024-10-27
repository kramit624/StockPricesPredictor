from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Load the trained model
model = LinearRegression()

# Function to create and train the model
def create_model():
    data = pd.read_csv('stock_prices.csv')  # Ensure you have this CSV file
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal
    
    # Prepare features and labels
    features = data[['Date', 'Open', 'High', 'Low', 'Volume']]
    labels = data['Close']
    
    model.fit(features, labels)  # Train the model
    return model

# Call the model creation function
create_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<ticker>', methods=['GET'])
def predict(ticker):
    try:
        # Fetch stock data
        stock_data = yf.download(ticker, period="1y")
        
        if stock_data.empty:
            return jsonify({'error': f'No data found for ticker: {ticker}'}), 404

        # Prepare features
        stock_data['Date'] = pd.to_datetime(stock_data.index)
        stock_data['Date'] = stock_data['Date'].map(lambda x: x.toordinal())
        
        # Ensure all features are included
        features = stock_data[['Date', 'Open', 'High', 'Low', 'Volume']]
        
        # Make predictions
        predictions = model.predict(features)
        
        # Get the last prediction (for example, the most recent)
        predicted_price = predictions[-1] if len(predictions) > 0 else None

        # Plot and save the prediction graph
        plt.figure(figsize=(10, 5))
        plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Price', color='blue')
        plt.plot(stock_data['Date'], predictions, label='Predicted Price', color='orange')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        
        # Save the figure in static/images
        image_path = f'./static/images/{ticker}_prediction.png'
        plt.savefig(image_path)
        plt.close()

        # Return prediction as JSON
        return jsonify({'ticker': ticker, 'predicted_price': predicted_price, 'image': image_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
