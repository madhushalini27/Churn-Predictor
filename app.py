#Gender  => 1 Female   0 MAle
#Churn  => 1 yes 0 Nos
#Scaler is exported as scaler.pkl
#Model is exported as model.ProcessLookupError
#order of the X =>'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import numpy as np
import joblib

scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")

st.title("Churn Predictor App")

st.divider()

st.write("Please enter values and hit the predict button for getting an prediction")

st.divider()

age=st.number_input("Enter age",min_value=10,max_value=100,value=30)

tenure=st.number_input("Enter the tenure",min_value=0,max_value=130,value=10)

monthlycharge=st.number_input("enter the monthly charge ",min_value=30,max_value=150)

gender=st.selectbox("Enter the Gender",["Male","Female"])

st.divider()

predictbutton=st.button("Predict")

if predictbutton:
         gender_selected=1 if gender=="Female" else 0
         X=[age, gender_selected, tenure, monthlycharge]
         x1=np.array(X)
         X_array=scaler.transform([x1])
         prediction=model.predict(X_array)[0]
         predicted ="Yes" if prediction ==1 else "No"
         st.write(f"Predicted :{predicted}")

else:
        st.write("Please enter the values and use predict button")


