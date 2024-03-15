from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load("E:/fast_api/best_model.pkl")

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON request
        data = request.json
        
        # Validate input data
        if not all(key in data for key in ('carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z')):
            return jsonify({'error': 'Missing data'}), 400
        
        # Extract features from input data
        carat = float(data['carat'])
        cut = data['cut']
        color = data['color']
        clarity = data['clarity']
        depth = float(data['depth'])
        table = float(data['table'])
        x = float(data['x'])
        y = float(data['y'])
        z = float(data['z'])
        
        # Convert categorical features into one-hot encoding
        cut_list = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
        color_list = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
        clarity_list = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
        
        cut_idx = cut_list.index(cut)
        color_idx = color_list.index(color)
        clarity_idx = clarity_list.index(clarity)
        
        # Construct feature vector
        features = np.array([carat, cut_idx, color_idx, clarity_idx, depth, table, x, y, z]).reshape(1, -1)
        
        # Predict using the model
        predicted_price = model.predict(features)[0]
        
        # Print received data and predicted price
        print("Received data:", data)
        print("Predicted price:", predicted_price)
        
        # Return prediction
        return jsonify({'predicted_price': predicted_price}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


