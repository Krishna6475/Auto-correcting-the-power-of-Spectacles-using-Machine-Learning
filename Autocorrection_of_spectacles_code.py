import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import time

# Part 1: Simulate sensor data and save it
def generate_sensor_data(num_samples):
    np.random.seed(42)  # For reproducibility
    focal_lengths = np.linspace(1.0, 3.0, num_samples)  # Example focal lengths
    clarity_scores = np.exp(-((focal_lengths - 2.0) ** 2)) + 0.1 * np.random.randn(num_samples)  # Add some noise
    return focal_lengths, clarity_scores

num_samples = 1000
focal_lengths, clarity_scores = generate_sensor_data(num_samples)

# Create a DataFrame for the data
data = pd.DataFrame({
    'focal_length': focal_lengths,
    'clarity_score': clarity_scores
})

# Save the data to a CSV file
data.to_csv('sensor_data.csv', index=False)

# Part 2: Load data, train the model, and save it
# Load the data
data = pd.read_csv('sensor_data.csv')

# Split the data into training and testing sets
X = data[['focal_length']]
y = data['clarity_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
joblib.dump(model, 'focal_length_model.pkl')

# Part 3: Real-time adjustment functions
def read_sensor_data():
    # Simulated sensor data: read the current clarity score
    current_focal_length = np.random.uniform(1.0, 3.0)
    clarity_score = np.exp(-((current_focal_length - 2.0) ** 2)) + 0.1 * np.random.randn()
    return current_focal_length, clarity_score

def adjust_lens(predicted_focal_length):
    print(f'Adjusting lens to focal length: {predicted_focal_length:.2f}')

# Real-time adjustment loop
def real_time_adjustment():
    model = joblib.load('focal_length_model.pkl')  # Load the trained model
    while True:
        current_focal_length, clarity_score = read_sensor_data()
        print(f'Current focal length: {current_focal_length:.2f}, Clarity score: {clarity_score:.2f}')
        
        # Predict the best focal length
        predicted_focal_length = model.predict([[clarity_score]])
        adjust_lens(predicted_focal_length[0])
        
        # Simulate a short delay
        time.sleep(1)

# Part 4: User input for focal length and clarity score
def get_user_input():
    try:
        # Take user input for current focal length and clarity score
        current_focal_length = float(input("Enter the current focal length (e.g., 1.5): "))
        clarity_score = float(input("Enter the current clarity score (e.g., 0.8): "))
        return current_focal_length, clarity_score
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return get_user_input()

def predict_adjustable_focal_length(model, clarity_score):
    # Predict the best focal length based on clarity score
    predicted_focal_length = model.predict([[clarity_score]])
    return predicted_focal_length[0]

def main():
    model = joblib.load('focal_length_model.pkl')  # Load the trained model
    current_focal_length, clarity_score = get_user_input()
    print(f'Current focal length: {current_focal_length:.2f}, Clarity score: {clarity_score:.2f}')
    
    # Predict the adjustable focal length
    predicted_focal_length = predict_adjustable_focal_length(model, clarity_score)
    print(f'Predicted adjustable focal length: {predicted_focal_length:.2f}')
    
    # Simulate the lens adjustment
    adjust_lens(predicted_focal_length)

if __name__ == "__main__":
    main()
