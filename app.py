
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

@st.cache_data
def load_metadata():
    with open("model_metadata.json", "r") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #17365D;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
}
div.stButton > button {
    background: #17365D;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏦 Loan Approval AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Intelligent Loan Eligibility Prediction System</p>',
    unsafe_allow_html=True
)

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric("AI Model", metadata["model_name"])
col2.metric("Test Accuracy", f'{metadata["accuracy"] * 100:.2f}%')
col3.metric("ROC AUC", f'{metadata["roc_auc"]:.4f}')

st.divider()

st.subheader("👤 Applicant Information")
st.write("Enter the applicant details below to generate a prediction.")

with st.form("loan_form"):

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

    with c2:
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=0.0
        )
        loan_amount = st.number_input(
            "Loan Amount (Thousands)",
            min_value=1.0,
            value=120.0
        )
        loan_term = st.selectbox(
            "Loan Term (Months)",
            [360, 180, 240, 120, 60, 36, 12]
        )
        credit_history = st.selectbox(
            "Credit History",
            ["Good (1)", "Poor (0)"]
        )
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

if submitted:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1.0 if credit_history == "Good (1)" else 0.0,
        "Property_Area": property_area
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("📊 Prediction Results")

    approval_probability = float(probability[1]) * 100
    rejection_probability = float(probability[0]) * 100

    if prediction == 1:
        st.success("✅ Predicted Result: LOAN APPROVED")
    else:
        st.error("❌ Predicted Result: LOAN REJECTED")

    r1, r2 = st.columns(2)

    r1.metric("Approval Probability", f"{approval_probability:.2f}%")
    r2.metric("Rejection Probability", f"{rejection_probability:.2f}%")

    st.progress(approval_probability / 100)

    st.caption(
        "Probabilities are model estimates, not guaranteed lending decisions."
    )

st.divider()

st.markdown("""
<div style="text-align:center;color:#64748b;">
    Loan Approval AI | Machine Learning Project
    <br>
    Built with Python, Scikit-learn and Streamlit
    <br><br>
    Educational project only. Not an actual banking decision.
</div>
""", unsafe_allow_html=True)
