import streamlit as st
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')
numeric_cols = joblib.load('numeric_cols.pkl')

st.title("Credit Risk Prediction")
st.write("Enter applicant details to predict credit risk.")

# --- Input form ---
st.header("Applicant Information")

duration = st.slider("Loan Duration (months)", 4, 72, 24)
credit_amount = st.slider("Credit Amount (DM)", 250, 20000, 3000)
age = st.slider("Age", 18, 75, 35)
installment_rate = st.slider("Installment Rate (% of income)", 1, 4, 2)
residence_since = st.slider("Years at Current Residence", 1, 4, 2)
existing_credits = st.slider("Existing Credits at Bank", 1, 4, 1)
num_dependents = st.slider("Number of Dependents", 1, 2, 1)

checking_status = st.selectbox(
    "Checking Account Status",
    options=['A11', 'A12', 'A13', 'A14'],
    format_func=lambda x: {
        'A11': '< 0 DM',
        'A12': '0 - 200 DM',
        'A13': '>= 200 DM',
        'A14': 'No checking account'
    }[x]
)

credit_history = st.selectbox(
    "Credit History",
    options=['A30', 'A31', 'A32', 'A33', 'A34'],
    format_func=lambda x: {
        'A30': 'No credits taken',
        'A31': 'All credits paid back duly',
        'A32': 'Existing credits paid back duly',
        'A33': 'Delay in past payments',
        'A34': 'Critical account / other credits'
    }[x]
)

purpose = st.selectbox(
    "Loan Purpose",
    options=['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
    format_func=lambda x: {
        'A40': 'New car', 'A41': 'Used car', 'A42': 'Furniture/equipment',
        'A43': 'Radio/TV', 'A44': 'Domestic appliances', 'A45': 'Repairs',
        'A46': 'Education', 'A48': 'Retraining', 'A49': 'Business', 'A410': 'Other'
    }[x]
)

savings_status = st.selectbox(
    "Savings Account Status",
    options=['A61', 'A62', 'A63', 'A64', 'A65'],
    format_func=lambda x: {
        'A61': '< 100 DM', 'A62': '100-500 DM', 'A63': '500-1000 DM',
        'A64': '>= 1000 DM', 'A65': 'Unknown/no savings'
    }[x]
)

employment = st.selectbox(
    "Employment Duration",
    options=['A71', 'A72', 'A73', 'A74', 'A75'],
    format_func=lambda x: {
        'A71': 'Unemployed', 'A72': '< 1 year', 'A73': '1-4 years',
        'A74': '4-7 years', 'A75': '>= 7 years'
    }[x]
)

housing = st.selectbox(
    "Housing",
    options=['A151', 'A152', 'A153'],
    format_func=lambda x: {'A151': 'Rent', 'A152': 'Own', 'A153': 'Free'}[x]
)

job = st.selectbox(
    "Job",
    options=['A171', 'A172', 'A173', 'A174'],
    format_func=lambda x: {
        'A171': 'Unemployed/unskilled', 'A172': 'Unskilled resident',
        'A173': 'Skilled employee', 'A174': 'Management/self-employed'
    }[x]
)

foreign_worker = st.selectbox("Foreign Worker", options=['A201', 'A202'],
    format_func=lambda x: {'A201': 'Yes', 'A202': 'No'}[x])

personal_status = st.selectbox(
    "Personal Status",
    options=['A91', 'A92', 'A93', 'A94'],
    format_func=lambda x: {
        'A91': 'Male: divorced/separated', 'A92': 'Female: divorced/separated/married',
        'A93': 'Male: single', 'A94': 'Male: married/widowed'
    }[x]
)

other_parties = st.selectbox(
    "Other Debtors/Guarantors",
    options=['A101', 'A102', 'A103'],
    format_func=lambda x: {'A101': 'None', 'A102': 'Co-applicant', 'A103': 'Guarantor'}[x]
)

property_magnitude = st.selectbox(
    "Property",
    options=['A121', 'A122', 'A123', 'A124'],
    format_func=lambda x: {
        'A121': 'Real estate', 'A122': 'Building society savings/life insurance',
        'A123': 'Car or other', 'A124': 'Unknown/no property'
    }[x]
)

other_payment_plans = st.selectbox(
    "Other Installment Plans",
    options=['A141', 'A142', 'A143'],
    format_func=lambda x: {'A141': 'Bank', 'A142': 'Stores', 'A143': 'None'}[x]
)

own_telephone = st.selectbox("Owns Telephone", options=['A191', 'A192'],
    format_func=lambda x: {'A191': 'No', 'A192': 'Yes'}[x])

if st.button("Predict Credit Risk"):
    input_dict = {
        'checking_status': checking_status, 'duration': duration,
        'credit_history': credit_history, 'purpose': purpose,
        'credit_amount': credit_amount, 'savings_status': savings_status,
        'employment': employment, 'installment_rate': installment_rate,
        'personal_status': personal_status, 'other_parties': other_parties,
        'residence_since': residence_since, 'property_magnitude': property_magnitude,
        'age': age, 'other_payment_plans': other_payment_plans,
        'housing': housing, 'existing_credits': existing_credits,
        'job': job, 'num_dependents': num_dependents,
        'own_telephone': own_telephone, 'foreign_worker': foreign_worker
    }

    input_df = pd.DataFrame([input_dict])

    categorical_cols_input = input_df.select_dtypes(include='object').columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols_input)

    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

    proba = model.predict_proba(input_encoded)[:, 1][0]
    prediction = "Bad Credit Risk" if proba >= 0.4 else "Good Credit"

    st.subheader("Prediction Result")
    st.write(f"**{prediction}**")
    st.write(f"Risk probability: {proba:.2%}")

    st.subheader("What drives this model's decisions")
    importances = pd.Series(model.feature_importances_, index=model_columns)
    top_features = importances.sort_values(ascending=False).head(8)
    st.bar_chart(top_features)
    st.caption("These are the features the model relies on most heavily across all predictions, not specific to this applicant.")
