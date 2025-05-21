import streamlit as st
import pandas as pd

def app():
    st.title('Data Upload')
    file = st.file_uploader('Upload CSV', type=['csv','xlsx'])
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.session_state['df']=df
        st.dataframe(df.head())