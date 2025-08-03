
# Fraud Detection Project

A machine learning-based system to predict fraudulent financial transactions using classification models and cost-sensitive threshold tuning. Deployed as a Streamlit web app using the best-performing model.

---

## Project Structure

```
fraud-detection/
├── data/            # Raw and processed datasets
├── models/          # Trained model .pkl files
├── notebooks/       # EDA, training, threshold tuning, comparison notebooks
├── test_data/       # Stored test sets
├── app/             # Streamlit app folder
│   └── fraud.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Features

- Used baseline models like XGBoost, Light GBM, Logistic regression for classification.
- Used sampling techniques like SMOTE, ADASYN, oversampling the minority for class imbalance.
- Tuned threshold for cost optimization. In fraud detection, false negatives are more costly than false positives(cost = 10000 for false negatives and 100 for false positives).
- Tuned threshold for F1 score optimization.
- Used random splitting as well as time based split into test and train.
- A detailed comparison for all the combinations of models, sampling techniques, splitting techniques and threshold tuning technique is made in comparison.ipynb
- Deployed a global app using streamlit to run the app using best performing model(Light GBM + SMOTE + cost based optimization).


---

## How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run Streamlit app:**
```bash
cd app
streamlit run fraud.py
```

---

## Model Performance - Cost-Based Threshold Tuning

| Model                 | Best Threshold | Precision | Recall  | F1 Score | Cost      |
|----------------------|----------------|-----------|---------|----------|-----------|
| XGBoost + SMOTE      | 0.7920         | 0.4620    | 0.9921  | 0.6304   | ₹319,800  |
| XGBoost + ADASYN     | 0.9009         | 0.4563    | 0.9915  | 0.6250   | ₹334,100  |
| LightGBM + SMOTE     | 0.9009         | 0.4303    | 0.9915  | 0.6001   | ₹355,700  |
| LightGBM + ADASYN    | 0.9009         | 0.4047    | 0.9951  | 0.5754   | ₹320,500  |
| Logistic + SMOTE     | 0.7920         | 0.0570    | 0.8584  | 0.1070   | ₹6,987,200|
| Time_XGB + SMOTE     | 0.4852         | 0.8221    | 0.9976  | 0.9014   | ₹75,700   |
| Time_XGB + ADASYN    | 0.6732         | 0.8101    | 0.9982  | 0.8944   | ₹68,700   |
| Time_LGBM + SMOTE    | 0.8514         | 0.8571    | 0.9976  | 0.9220   | ₹67,500   |
| Time_LGBM + ADASYN   | 0.8910         | 0.8418    | 0.9976  | 0.9131   | ₹71,000   |

---

## Model Performance - F1 Score-Based Threshold Tuning

| Model                 | Threshold | Precision | Recall  | F1 Score | FP    | FN   | Cost      |
|----------------------|-----------|-----------|---------|----------|-------|------|-----------|
| XGBoost + SMOTE      | 0.99      | 0.8008    | 0.8929  | 0.8443   | 365   | 176  | ₹1,796,500 |
| XGBoost + ADASYN     | 0.99      | 0.5903    | 0.9148  | 0.7176   | 1043  | 140  | ₹1,504,300 |
| LightGBM + SMOTE     | 0.99      | 0.8688    | 0.7821  | 0.8232   | 194   | 358  | ₹3,599,400 |
| LightGBM + ADASYN    | 0.94      | 0.4222    | 0.9836  | 0.5908   | 2212  | 27   | ₹491,200   |
| Logistic + SMOTE     | 0.99      | 0.3321    | 0.6209  | 0.4328   | 3077  | 934  | ₹9,647,700 |
| Time_XGB + SMOTE     | 0.97      | 0.9292    | 0.9764  | 0.9522   | 123   | 39   | ₹402,300   |
| Time_XGB + ADASYN    | 0.96      | 0.8990    | 0.9843  | 0.9397   | 183   | 26   | ₹278,300   |
| Time_LGBM + SMOTE    | 0.96      | 0.9220    | 0.9867  | 0.9533   | 138   | 22   | ₹233,800   |
| Time_LGBM + ADASYN   | 0.91      | 0.8559    | 0.9873  | 0.9169   | 275   | 21   | ₹237,500   |

---

## Live Streamlit App

You can try the deployed model through the Streamlit interface to simulate real-time fraud checks:

>  [Live App](https://your-app-name.streamlit.app)  


---

##  Requirements

- `pandas`, `numpy`
- `scikit-learn`, `imbalanced-learn`
- `lightgbm`, `xgboost`
- `streamlit`
- `joblib`

---

##  Author

**Ayush Yadav**  
MTech, IIT Bombay  

---

##  License

This project is licensed under the [MIT License](LICENSE).
