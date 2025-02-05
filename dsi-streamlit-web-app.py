import joblib
import pandas as pd
import streamlit as st


# load the model pipeline
model = joblib.load("model.joblib")

# add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")


# age input form
age = st.number_input(
    "01. Enter the Customer's Age", 
    min_value=18, 
    max_value=120,
    value=35)


# gender input
gender = st.radio(
    label="02. Enter the Customer's Gender",
    options=["M", "F"])


# credit score input
credit_score = st.number_input(
    "03. Enter the Customer's Credit Score", 
    min_value=0, 
    max_value=1000,
    value=500)


# submit input to model
if st.button("Submit for Prediction"):
    
    # store data in a df for prediction
    new_data = pd.DataFrame({"age": [age], "gender": [gender], "credit_score": credit_score})
    
    
    # apply input to pipeline
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # display the output prediction
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")








