import streamlit as st
import analysis_tab
import forecasting_tab
import sql_query_tab  # new import

st.set_page_config(page_title="THC Forecasting App", layout="wide")
st.title("📈 THC Sales Forecasting Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔮 Forecasting", "🧠 SQL Query"])  # add tab3

with tab1:
    analysis_tab.run()

with tab2:
    forecasting_tab.run()

with tab3:
    sql_query_tab.run()  # new SQL query interface
