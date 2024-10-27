import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('stock_prices.csv')  # Ensure you have this CSV file
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal

# Prepare features and labels
features = data[['Date', 'Open', 'High', 'Low', 'Volume']]
labels = data['Close']

# Create and train the model
model = LinearRegression()
model.fit(features, labels)

# Optionally, save the model using joblib or pickle (if needed)
# import joblib
# joblib.dump(model, 'stock_price_model.pkl')
