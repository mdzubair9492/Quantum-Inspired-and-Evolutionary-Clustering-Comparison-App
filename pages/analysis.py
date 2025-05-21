import streamlit as st
from algorithms.kmeans import fit_predict as km_fp
from algorithms.ga_kmeans import fit_predict as ga_fp
from algorithms.qga_kmeans import fit_predict as qg_fp
from utils.plotting import plot_clusters

def app():
    st.title("Analysis")
    data = st.session_state.get("data")
    if data is None:
        st.warning("Please preprocess data first.")
        return

    k = st.slider("Number of clusters", 2, 10, 3)

    def run_and_store(name, func):
        
        res = func(data, {"n_clusters": k})
        if len(res) == 3:
            labels, metrics, _ = res
        else:
            labels, metrics = res
        metrics["algorithm"] = name
        st.session_state.metrics_store.append(metrics)

        st.subheader(f"{name} Clusters")
        st.write(metrics)
        fig = plot_clusters(data, labels, title=f"{name} Clusters")
        st.pyplot(fig)

    if st.button("Run Classical K-means"):
        run_and_store("K-means", km_fp)
    if st.button("Run GA K-means"):
        run_and_store("GA K-means", ga_fp)
    if st.button("Run QIGA K-means"):
        run_and_store("QIGA K-means", qg_fp)

    
    st.write("Debug — metrics_store:", st.session_state.metrics_store)







