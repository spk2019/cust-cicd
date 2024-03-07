import streamlit as st
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import joblib 

model = joblib.load('modelv1.joblib')
transformer = joblib.load('transformerv1.joblib')
st.title("Telco Customer Churn")

creditscore = st.number_input("Choose Credit Score",350,850)
geography = st.selectbox("Choose Country", ["Spain","Germany","France"])
gender = st.selectbox("Choose Sex", ["Male","Female"])
age = st.slider("Choose Age",18,100)
tenure = st.number_input("Choose Tenure",0,10)
balance = st.number_input("Choose Balance",0)
products = st.selectbox("Choose NumOfProducts", [1,2,3,4])
hasCrCard = st.selectbox("Choose HasCrCard", [0,1])
isActiveMember = st.selectbox("Choose IsActiveMember", [0,1])
estimatedSalary = st.number_input("Choose Estimated Salary")

columns = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

def predict(): 
    row = np.array([creditscore,geography,gender,age,tenure,balance,products,hasCrCard,isActiveMember,estimatedSalary]) 
    x = pd.DataFrame([row], columns = columns)
    print(x)
    
    X = transformer.transform(x)
    
    
    print(X.shape)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Customer is likely to leave :thumbsup:')
    else: 
        st.error('Customer is not likely to leave :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)