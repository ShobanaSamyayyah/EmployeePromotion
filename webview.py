import streamlit as st
import pandas as pd
import joblib

st.title("Predicting Promotion")
df = pd.read_csv("train.csv")


employee_id = st.number_input("employee_id")
no_of_trainings = st.selectbox('No: of Trainings',df['no_of_trainings'].unique())
age = st.number_input("age")
previous_year_rating = st.selectbox('Last Year Rating',df['previous_year_rating'].unique())
length_of_service = st.selectbox('Overall Experience',df['length_of_service'].unique())

kPI_met = st.selectbox('Education Level',df['KPIs_met >80%'].unique())
award_won = st.selectbox('Award Won',df['awards_won?'].unique())
avg_training_score = st.selectbox('Avg. Training Score',df['avg_training_score'].unique())
department = st.selectbox('Department',df['department'].unique())
region = st.selectbox('Region',df['region'].unique())
education = st.selectbox('Education',df['education'].unique())
gender = st.selectbox('Gender',df['gender'].unique())
recruitment_channel = st.selectbox('Recruitment Channel',df['recruitment_channel'].unique())

inputs = {
  
    "no_of_trainings": no_of_trainings,
    "age": age,
    "previous_year_rating": previous_year_rating,
    "length_of_service": length_of_service,
    "KPIs_met >80%": kPI_met,
    "awards_won?": award_won,
    "avg_training_score": avg_training_score,
    "department": department,
    "region": region,
    "education": education,
    "gender": gender,
    "recruitment_channel": recruitment_channel
}
if(st.button("Predict")):
  model = joblib.load("XGB_Tune_0.39.pkl")
  X_inputs = pd.DataFrame([inputs])
  prediction = model.predict(X_inputs)
  st.write(prediction)
