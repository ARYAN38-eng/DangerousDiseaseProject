import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder 
import pandas as pd
import pickle 
df=pd.read_csv("dangerous.csv")
model=pickle.load(open('dangerous_rfc.pkl','rb'))
ohe=pickle.load(open('dangerous_ohe.pkl','rb'))
st.title("Dangerous Disease Detection")
AnimalName=st.selectbox("Animal Name",sorted(df['AnimalName'].unique()))
symptoms1=st.selectbox("Symptoms1",sorted(df['symptoms1'].unique()))
symptoms2=st.selectbox("Symptoms2",sorted(df['symptoms2'].unique()))
symptoms3=st.selectbox("Symptoms3",sorted(df['symptoms3'].unique()))
symptoms4=st.selectbox("Symptoms4",sorted(df['symptoms4'].unique()))
symptoms5=st.selectbox("Symptoms5",sorted(df['symptoms5'].unique()))

def detect():
    query=pd.DataFrame({
        'AnimalName':[AnimalName],
        'symptoms1':[symptoms1],
        'symptoms2':[symptoms2],
        'symptoms3':[symptoms3],
        'symptoms4':[symptoms4],
        'symptoms5':[symptoms5]
    })
    transformed_query=ohe.transform(query)
    Detection=model.predict(transformed_query)
    return Detection
    
if st.button("Detect"):
    Detect=detect()
    if Detect[0]==1:
        st.write('The Disease is Dangerous')
    else:
        st.write("The Disease is not Dangerous")
        
    
    
