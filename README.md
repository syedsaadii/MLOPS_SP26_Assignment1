# Diabetes Prediction using MLOps Pipeline

This project is an end-to-end Machine Learning Operations (MLOps) implementation for diabetes prediction using a medical dataset. The project includes data preprocessing, exploratory data analysis (EDA), machine learning model training, model evaluation, FastAPI deployment, API testing, and version control using GitHub.

---

# Project Overview

The main objective of this project is to build a machine learning model capable of predicting whether a patient is diabetic or non-diabetic based on medical attributes such as:

- Age
- BMI
- HbA1c
- Cholesterol
- Urea
- Creatinine
- HDL / LDL
- Triglycerides
- Gender

The trained model is deployed using FastAPI and can be accessed through REST API endpoints.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- FastAPI
- Pydantic
- Uvicorn
- Joblib
- Git & GitHub

---

# Dataset

Dataset used:

`diabetes_unclean.csv`

The dataset contains patient medical records used for diabetes prediction.

---

# Project Workflow

## 1. Data Cleaning & Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns:
  - `ID`
  - `No_Pation`
- Corrected invalid gender values
- Handled missing values
- Applied one-hot encoding on categorical columns
- Prepared data for machine learning models

---

## 2. Exploratory Data Analysis (EDA)

The following visualizations were created:

- Gender distribution bar chart
- Age distribution histogram
- BMI distribution histogram
- BMI vs HbA1c scatter plot
- Age vs HbA1c scatter plot
- BMI boxplot for diabetic vs non-diabetic patients

---

# Machine Learning Models

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

---

# Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

The best-performing model was selected and saved using Joblib.

---

# Saved Files

The following files are generated after model training:

- `diabetes_model.pkl`
- `training_columns.pkl`

---

# FastAPI Deployment

The machine learning model was deployed using FastAPI.

## API Features

- Input validation using Pydantic
- REST API endpoint for predictions
- Automatic Swagger documentation
- Error handling and validation

---

# API Endpoints

## Health Check Endpoint

### GET /

Returns API running status.

### Example Response

```json
{
  "status": "API is running"
}
