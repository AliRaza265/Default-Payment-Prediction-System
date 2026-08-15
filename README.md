# Default Payment Prediction System

A Machine Learning classification project that predicts whether a credit card customer is likely to default on their next payment.

## 📌 Project Overview

The Default Payment Prediction System uses customer financial and payment-related information to predict the possibility of a customer defaulting on their next credit card payment.

This project was developed as a practical Machine Learning project to apply data preprocessing, class balancing, model training, evaluation, and prediction techniques.

## 🎯 Project Objective

The main objective of this project is to:

- Analyze customer credit and payment information
- Handle an imbalanced classification dataset
- Predict whether a customer will default on their next payment
- Compare different classification algorithms
- Select a suitable final model
- Provide predictions through a simple user interface

## 📊 Dataset

The project uses the **Default of Credit Card Clients** dataset.

The original dataset contains approximately 30,000 customer records.

For this practice project, **3,000 records** were selected from the original dataset to keep the project manageable on the available hardware.

### Target Variable

`default.payment.next.month`

| Value | Meaning |
|---|---|
| `0` | Customer does not default |
| `1` | Customer defaults |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Streamlit
- Pickle

## 🔄 Machine Learning Workflow

The project follows this workflow:

1. Load the dataset
2. Select a sample of 3,000 records
3. Understand the dataset
4. Check missing values and duplicates
5. Remove selected unnecessary features
6. Separate features and target
7. Split data into training and testing sets
8. Handle class imbalance using **SMOTE**
9. Scale the features using **MinMaxScaler**
10. Train classification models
11. Compare model performance
12. Select the final model
13. Save the trained model and preprocessing objects
14. Use the model for new predictions

## 🤖 Models Tested

The following classification models were tested:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Voting Classifier

After comparing the models, **Random Forest Classifier** was selected as the final model.

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Because the dataset contains an imbalanced target, special attention was given to the performance of the **default class (1)** rather than relying only on accuracy.

## 🌲 Final Model

The final model selected for the project is:

**Random Forest Classifier**

The model was selected after comparing multiple classification approaches and considering their overall classification performance.

## 🖥️ Application

A simple Streamlit application is included in the project.

The application allows a user to enter customer information and receive a prediction indicating whether the customer is likely to default on their next payment.

### Prediction Output

The system provides one of two predictions:

- **Customer is likely to default**
- **Customer is not likely to default**

## 📁 Project Structure

```text
Default-Payment-Prediction-System/
│
├── data_set/
│   └── UCI_Credit_Card.csv
│
├── Models/
│   ├── Model.pkl
│   ├── Scaler.pkl
│   └── Encoder.pkl
│
├── main.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
