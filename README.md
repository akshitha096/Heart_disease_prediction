# Heart_disease_prediction
# BeWell Heart Disease Prediction System

The **BeWell Heart Disease Prediction System** is a machine learning-based application designed to predict the likelihood of heart disease in individuals based on various health-related features. It uses historical health data to train machine learning models that can classify individuals as "at risk" or "not at risk" for heart disease.

This project includes a graphical user interface (GUI) for users to interact with the system, upload data, and receive predictions.

## Features

- **Data Upload**: Users can upload health data for prediction.
- **Heart Disease Prediction**: The system uses machine learning models to predict if an individual is at risk for heart disease.
- **Model Training**: Supports training machine learning models such as Logistic Regression, Random Forest, and Support Vector Machine (SVM).
- **Visualization**: Provides graphical representation of model performance, including accuracy and confusion matrix.
- **Real-Time Prediction**: Predict heart disease based on user-provided data.

## Technologies Used

- **Python**: The programming language used for implementing the system.
- **Tkinter**: GUI library to create the user interface.
- **Scikit-learn**: Used for machine learning algorithms like Logistic Regression, Random Forest, and SVM.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting performance graphs and confusion matrices.
- **Seaborn**: For advanced data visualization.

## Installation

### Prerequisites

Make sure you have Python installed, and then you can install the required dependencies using `pip`:
pip install tkinter
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn

### Model Evaluation
The system evaluates the trained model on accuracy, precision, recall, and F1-score. It also displays a confusion matrix to give insights into the model's performance on test data.

