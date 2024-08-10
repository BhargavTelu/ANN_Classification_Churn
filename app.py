import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
from tensorflow.keras.models import load_model

# load the trained model and scaler and encoders
model = load_model("model.h5")
with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)
with open("dummy_geo.pkl","rb") as file:
    dummy_geo = pickle.load(file)      


# user inputs
st.title("customer churn prediction")

geography= st.selectbox("Geography",dummy_geo.categories_[0])
gender=st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance  = st.number_input("Balance")
CreditScore= st.number_input("Credit Score")
EstimatedSalary = st.number_input("Estimated Salary")
Tenure = st.slider("Tenure",0,10)
NumOfProducts = st.slider("Num Of Products",1,4)
HasCrCard = st.selectbox("Has Credit Card",[0,1])
IsActiveMember = st.selectbox("Is Active Member",[0,1])


# preparing the input data
input_data = pd.DataFrame({
    "CreditScore" : [CreditScore],
    "Gender" : [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [Tenure],
    "Balance": [balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [HasCrCard],
    "IsActiveMember": [IsActiveMember],
    "EstimatedSalary": [EstimatedSalary]
})


geo_encoded = dummy_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns = dummy_geo.get_feature_names_out(["Geography"]))

# concatenating df and geography
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# scaling the input data
input_data_scaled = scaler.transform(input_data)

#predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f"Churn Probability: {prediction_prob:.2f}")

if prediction_prob>0.5:
    st.write("The person is likly to churn")
else:
    st.write("The person is not likly to churn")    