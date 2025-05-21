import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def plot_clusters(data: np.ndarray, labels: np.ndarray, title: str = 'Clusters') -> plt.Figure:

    fig, ax = plt.subplots()
    sc = ax.scatter(data[:, 0], data[:, 1], c=labels)
    ax.set_title(title)
    return fig