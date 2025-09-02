# Churn Analysis & Prediction – Machine Learning

A Python-based project to analyze customer churn data and build a Flask web application showing predictions using an ensemble of models (Logistic Regression, Decision Tree, Random Forest).

## Project Structure
ChurnAnalysis_MachineLearning/
│
├── Dataset/
│   └── tel_churn.csv              ← Telco customer data
│
├── Churn_Analysis- EDA.ipynb      ← Exploratory Data Analysis notebook
├── model_building.ipynb           ← Model training and evaluation
├── Churn_Prediction_Model.ipynb   ← Final model pipelines and export
│
├── models_pipeline.pkl            ← Pickled dict of all three trained models
├── app.py                         ← Flask application for predictions
│
└── templates/
    └── index.html                 ← Front-end form and results display

## Project Overview

Goal: Predict customer churn using features like tenure, services used, contract type, and more.

Data: Telco dataset including customer demographics, services status, and tenure.

## Approach:

Performed Exploratory Data Analysis with charts and statistics.

Built classification pipelines featuring One-Hot Encoding, Standard Scaling, and three models: Logistic Regression, Decision Tree, Random Forest.

Tuned models and evaluated performance using accuracy, ROC-AUC, and confidence scores.

Deployment: Created a Flask web app to accept user inputs and provide churn predictions and confidence percentages from each model.

## How to Run
1. Clone the repo
git clone https://github.com/Imam-Hussain-B/ChurnAnalysis_MachineLearning.git
cd ChurnAnalysis_MachineLearning

2. Install dependencies
pip install flask pandas scikit-learn

3. Launch the application
python app.py


Then open your browser at http://127.0.0.1:5000 — input the customer features and see churn predictions from all three models.

## Features

Interactive Flask UI to collect customer data and display results.

Preprocessing consistency using appended DataFrame + dummy encoding ensures inputs match model training.

Transparent predictions: model output, confidence level, and comparison between classifiers.

## Future Enhancements

Retrain using a single pipeline per model (preprocessing + classifier) stored via pickle for cleaner inference.

Add model performance metrics (e.g., ROC curve, precision/recall) on the UI.

Turn into an API for batch processing or integrate with messaging platforms.

## Contact & Contributions

For questions or collaboration opportunities:

Author: Imam Hussain

Repo: ChurnAnalysis_MachineLearning

Feel free to open an issue or PR—contributions and feedback are welcome!
