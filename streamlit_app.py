import streamlit as st
# riya's section  : UI & Visualization 

if "metrics_store" not in st.session_state:
    st.session_state.metrics_store = []

from pages import data_upload, preprocessing, analysis, comparison

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["Data Upload", "Preprocessing", "Analysis", "Comparison"]
)

if page == "Data Upload":
    data_upload.app()
elif page == "Preprocessing":
    preprocessing.app()
elif page == "Analysis":
    analysis.app()
elif page == "Comparison":
    comparison.app()


