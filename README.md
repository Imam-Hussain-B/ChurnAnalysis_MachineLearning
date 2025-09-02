## 📊 Customer Churn Analysis & Prediction (Machine Learning + Flask Deployment)

This project predicts whether a customer will churn (leave the service) or continue using a subscription-based service.

We use 3 machine learning models – Logistic Regression, Decision Tree, and Random Forest – and deploy them in a Flask web app for real-time predictions.

## 📂 Project Structure
ChurnAnalysis_MachineLearning/

│
├── Dataset/                       
│   └── tel_churn.csv              # Telco customer churn dataset 

(raw data)
│
├── templates/                     
│   └── index.html                 # HTML form (user input + prediction results)

│
├── app.py                         # Flask application (runs the web app)

├── models_pipeline.pkl            # Pickled dictionary containing 

3 trained models
│
├── notebooks/                     # Jupyter notebooks (for development & analysis)
│   ├── Churn_Analysis-EDA.ipynb   # Data cleaning, 

preprocessing, visualizations
│   ├── model_building.ipynb      # Training & evaluating ML 

models
│   └── Churn_Prediction_Model.ipynb # Finalized models + pickle 

creation
│
└── README.md             # Documentation (this file)

# 🚀 Workflow Overview

## Data Exploration (EDA)

Loaded Telco Customer Churn dataset.

Cleaned missing values, grouped tenure, handled categorical features.

Visualized churn distribution & key factors (gender, contract, payment method).

## Model Training

Encoded categorical variables (One-Hot Encoding).

Trained Logistic Regression, Decision Tree, and Random Forest.

Evaluated using Accuracy, ROC-AUC, Precision, Recall.

## Model Saving

Stored all 3 models in one .pkl file as a dictionary:

models = {
  "logistic": logistic_model,
  "decision_tree": dt_model,
  "random_forest": rf_model
}


Saved using pickle.dump(models, file)

Flask Deployment

Created app.py Flask app.

Users enter customer details via index.html form.

Models predict churn likelihood → Web app displays results with confidence.

# ⚙️ Installation & Running the App
1. Clone the repository
git clone https://github.com/Imam-Hussain-B/ChurnAnalysis_MachineLearning.git
cd ChurnAnalysis_MachineLearning

2. Install dependencies
pip install flask pandas scikit-learn

3. Run Flask app
python app.py

4. Open in browser

Go to: http://127.0.0.1:5000

🌐 Web App Flow

User enters details such as gender, contract type, monthly charges, tenure etc.

Flask processes the input → converts to model-compatible format.

All 3 models predict churn probability.

Results are shown like:

✅ Logistic Regression → “Likely to Continue” (Confidence: 82.5%)
⚠️ Decision Tree → “Likely to Churn” (Confidence: 70.1%)
🌳 Random Forest → “Likely to Continue” (Confidence: 88.3%)

# ✨ Features

📊 EDA + Visualizations to understand churn behavior.

🤖 Three machine learning models for prediction.

🌐 Flask deployment for real-time inference.

🎯 Confidence scores with every prediction.

# 🔮 Future Improvements

Integrate end-to-end pipeline preprocessing into pickle for cleaner inference.

Add graphs of churn probability in web UI.

Deploy on Heroku / AWS / Streamlit Cloud for public access.

Use XGBoost / LightGBM for better accuracy.

👨‍💻 Author

Imam Hussain
📌 GitHub Repository
