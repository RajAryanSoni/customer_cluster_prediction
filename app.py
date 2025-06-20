import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st


kmeans =joblib.load("Model.pkl")

df = pd.read_csv("Mall_Customers.csv")
x= df[["Annual Income (k$)", "Spending Score (1-100)"]]
x_array = x.values

#streamlit application page
st.set_page_config(page_title="Customer Cluster Prediction", layout="centered")
st.write("Customer cluster prediction")
st.write("Enter the customer Annual Income")


#inputs
annual_income=st.number_input("Annual income of a customer",min_value=0,max_value=400,value=50)
spending_score=st.slider("Spending Score between 1-100",1,100,20)



#predict the cluster
if st.button("Predict clusters"):
    input_data = np.array([[annual_income,spending_score]])
    cluster = kmeans.predict(input_data)
    st.success(f"Predicted cluster is :{cluster}")