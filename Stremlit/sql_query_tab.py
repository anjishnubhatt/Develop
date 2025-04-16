import streamlit as st
import pandas as pd
import sqlite3

def run():
    st.header("🧠 SQL Query on Forecast Data")

    if "best_forecasts" not in st.session_state:
        st.warning("Please run the Forecasting tab first to generate forecasts.")
        return

    df = st.session_state["best_forecasts"]

    st.markdown("### 🗃️ Preview Forecast Data")
    st.dataframe(df.head(), use_container_width=True)

    # Create in-memory SQLite DB
    conn = sqlite3.connect(":memory:")
    df.to_sql("forecast_data", conn, index=False, if_exists="replace")

    # SQL query input
    st.markdown("### 🛠️ Write SQL Query")
    default_query = "SELECT * FROM forecast_data LIMIT 10"
    query = st.text_area("Enter your SQL query below:", value=default_query, height=150)

    if st.button("Run Query"):
        try:
            result = pd.read_sql_query(query, conn)
            st.success("✅ Query executed successfully!")
            st.dataframe(result, use_container_width=True)

            st.download_button("📥 Download Results as CSV", result.to_csv(index=False).encode('utf-8'), "sql_query_results.csv")
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
