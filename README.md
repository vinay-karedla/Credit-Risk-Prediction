# Credit Risk Prediction

Predicts whether a loan applicant is a credit risk using the UCI German Credit dataset.

🔗 **Live Demo:** [https://credit-risk-prediction-ah3xrownqzkfj6crdpdz9c.streamlit.app/](https://credit-risk-prediction-ah3xrownqzkfj6crdpdz9c.streamlit.app/)

## Objective
Binary classification: good credit vs. bad credit risk.

## Approach
- EDA on 1,000 applicant records; identified 70/30 class imbalance and skewed numeric features.
- Preprocessing: one-hot encoding for categorical features, standardization for numeric features.
- Models: Logistic Regression (baseline), Random Forest (class_weight='balanced').
- Threshold tuning: default 0.5 threshold rejected in favor of 0.4, balancing recall and precision for bad-credit detection (missing a defaulter is costlier than a false alarm).

## Results (Random Forest, threshold=0.4)
- ROC-AUC: 0.79
- Recall (bad credit): 0.53
- Precision (bad credit): 0.53

## Key Risk Drivers
Credit amount, loan duration, age, and checking account status were the top predictive features.

## Interactive App
A Streamlit front end lets users input applicant details and get a live prediction, along with a feature-importance breakdown of what drives the model's decisions.

## Dataset
[UCI German Credit Data](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)

## Tech Stack
Python, pandas, scikit-learn, Streamlit
