# Auto-correcting-the-power-of-Spectacles-using-Machine-Learning

This README.md will give users a comprehensive understanding of how to use, contribute to, and run the project.

# Auto-Adjustment of Spectacles Power

This project demonstrates the use of machine learning to automatically adjust the power of spectacles based on real-time sensor data. The system predicts the ideal focal length based on a clarity score, allowing for real-time adjustment of spectacles for optimal vision.

## Project Overview

The main objective of this project is to create a smart spectacle system that uses machine learning to analyze real-time data, such as the clarity score, and adjusts the spectacles' focal length accordingly. The system uses a **Linear Regression** model trained on simulated sensor data, which predicts the best focal length for clear vision. 

The application is built using **Streamlit** for the user interface and **joblib** for model loading and prediction. The app allows users to input their clarity score and get a predicted focal length that the system would adjust the spectacles to. 

## Features

- **Real-time focal length prediction**: Based on a clarity score input by the user.
- **Automatic adjustment**: The system simulates adjusting the spectacles to the predicted focal length.
- **Interactive UI**: Built with **Streamlit** to allow seamless user interaction.
- **Model loading**: Uses a pre-trained model to predict focal length adjustments based on clarity score input.

## Project Architecture

- **Streamlit**: Used to create the user interface.
- **Linear Regression Model**: Trained on synthetic data to predict focal lengths based on clarity scores.
- **joblib**: Used to load and save the trained model.

### Files and Directories

- `app.py`: Main script to run the Streamlit app.
- `focal_length_model.pkl`: Pre-trained Linear Regression model saved using joblib.
- `README.md`: Project documentation (this file).
- `requirements.txt`: Python dependencies for the project.

## Installation

To get started with this project, follow the steps below:

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/auto-adjustment-spectacles.git
cd auto-adjustment-spectacles

### Explanation of the Sections:

1. **Project Overview**: This section explains the core idea behind the project—auto-adjusting spectacles power based on clarity scores and machine learning predictions.
   
2. **Features**: Lists the major functionalities of the project such as real-time prediction, automatic adjustment, and the interactive interface.

3. **Project Architecture**: Explains the main technologies used in the project (Streamlit, Linear Regression, joblib).

4. **Installation**: Step-by-step instructions for setting up the project on your local machine. It includes cloning the repository and installing dependencies.

5. **Usage**: Explains how the user interacts with the Streamlit app to input data and get predictions.

6. **Model Training (Optional)**: If you want to retrain the model, this section explains how to do so.

7. **Technologies Used**: Highlights the tools and libraries used to build the project (Python, Streamlit, joblib, scikit-learn).

8. **Future Enhancements**: A list of possible future improvements that can be made to the project.

9. **Contributing**: Encourages others to contribute to the project.

10. **License**: If applicable, includes the type of license under which the project is distributed (MIT License in this case).

### How to Use This:

- **Replace `your-username`** in the `git clone` link with your actual GitHub username.
- **Add more details**: If your project has more features, datasets, or requirements, feel free to add additional sections.

Local URL: http://localhost:8501
Network URL: http://20.20.27.7:8501
