import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.title('CHURN PREDICTION')
st.write("""
Created by Ridhwan Muttaqien - HCK06


         
Use the sidebar to input customer data.
""")
@st.cache_data
def fetch_data():
    df = pd.read_csv('model.csv')
    return df

df = fetch_data()

number_ref = st.selectbox('Number of Referrals', df['Number of Referrals'].unique())
tenor = st.number_input('Tenure in Months', 0.0)
offer = st.selectbox('Offer', df['Offer'].unique())
internet = st.selectbox('Internet Service', df['Internet Service'].unique())
contract = st.selectbox('Contract', df['Contract'].unique())
payment = st.selectbox('Payment Method', df['Payment Method'].unique())
charges = st.number_input('Total Charges', 0.0)
revenue = st.number_input('Total Revenue', 0.0)

data = {
    'Number of Referrals': number_ref,
    'Tenure in Months': tenor,
    'Offer': offer,
    'Internet Service': internet,
    'Contract': contract,
    'Payment Method': payment,
    'Total Charges': charges,
    'Total Revenue': revenue
}
input = pd.DataFrame(data, index=[0])

st.subheader('Customer Data')
st.write(input)

load_model = joblib.load("churn_pred.pkl")

if st.button('Predict'):
    prediction = load_model.predict(input)

    if prediction == 1:
        prediction = 'Churned'
    else:
        prediction = 'Stayed'

    st.write('Based on Customer Data, the customer predicted: ')
    st.write(prediction)