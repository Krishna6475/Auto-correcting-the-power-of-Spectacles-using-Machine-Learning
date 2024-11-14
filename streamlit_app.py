import streamlit as st
import joblib

# Display title
st.title("Auto-Adjustment of Spectacles Power")

# Load the trained model
try:
    model = joblib.load('focal_length_model.pkl')  # Make sure focal_length_model.pkl is in the same directory
    st.write("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'focal_length_model.pkl' not found. Please check the file path.")

# User input for clarity score
clarity_score_input = st.number_input("Enter the clarity score (e.g., 0.8)", min_value=0.0, max_value=1.0, step=0.01)

# Display the inputs
st.write(f"Clarity Score: {clarity_score_input:.2f}")

# Input for current focal length
current_focal_length = st.number_input("Enter the current focal length (e.g., 1.5)", min_value=0.0, step=0.1)

# Display the current focal length
st.write(f"Current Focal Length: {current_focal_length:.2f}")

# Button to predict the adjusted focal length based on clarity score
if st.button("Predict Adjusted Focal Length"):
    if 'model' in locals():  # Ensure the model is loaded
        predicted_focal_length = model.predict([[clarity_score_input]])  # Make the prediction
        st.write(f"Predicted Adjusted Focal Length: {predicted_focal_length[0]:.2f}")

        # Simulate adjusting the lens to the predicted focal length
        st.write(f"Adjusting lens to focal length: {predicted_focal_length[0]:.2f}")
