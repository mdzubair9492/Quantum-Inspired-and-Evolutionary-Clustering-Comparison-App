import streamlit as st
import pandas as pd
import altair as alt

def app():
    st.title("Comparison")
    metrics_store = st.session_state.get("metrics_store", [])
    if not metrics_store:
        st.info("Run at least one algorithm in Analysis first.")
        return

    
    df = pd.DataFrame(metrics_store).set_index("algorithm")
    st.write("## Metrics Table")
    st.dataframe(df)

    
    df_long = df.reset_index().melt(
        id_vars="algorithm", var_name="metric", value_name="value"
    )

    
    chart = (
        alt.Chart(df_long)
           .mark_bar()
           .encode(
               x=alt.X("metric:N", title="Metric"),
               y=alt.Y("value:Q", title="Value"),
               color=alt.Color("algorithm:N", title="Algorithm"),
               xOffset="algorithm:N"          
           )
           .properties(
               width=600,
               height=400
           )
    )

    st.write("## Metrics Comparison")
    st.altair_chart(chart, use_container_width=True)

   
    if st.button("Clear Comparison"):
        st.session_state.metrics_store = []
        st.experimental_rerun()




