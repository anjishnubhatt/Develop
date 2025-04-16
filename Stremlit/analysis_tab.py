import streamlit as st
import pandas as pd
import numpy as np

def run():
    st.header("Upload Your Sales Data")
    uploaded_file = st.file_uploader("Upload CSV with columns: date, sales, item, store", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["date"])
            if df.empty or df.columns.size == 0:
                st.error("Uploaded file appears to be empty or has no readable columns.")
                st.stop()

            st.success("File uploaded successfully!")
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["df"] = df  # ✅ this line is critical

            st.markdown("### Preview of Uploaded Data")
            st.dataframe(df.head())

            # -----------------------------------
            # 🔎 Data Analysis & Validation
            # -----------------------------------
            st.markdown("## Data Analysis & Validation")
            pk_nulls = df[['item', 'store', 'date']].isnull().sum()
            dup_count = df.duplicated(subset=['item', 'store', 'date']).sum()
            sales_nulls = df['sales'].isnull().sum()

            if pk_nulls.sum() > 0:
                st.error(f"Null values in primary key columns:\n{pk_nulls}")
            else:
                st.success("No nulls in primary key columns")

            if dup_count > 0:
                st.error(f"Found {dup_count} duplicate rows based on item-store-date")
            else:
                st.success("No duplicate rows in primary key")

            if sales_nulls > 0:
                st.error(f"{sales_nulls} null values found in `sales` column")
            else:
                st.success("No nulls in `sales` column")

            # -----------------------------------
            # 📊 Pareto Analysis
            # -----------------------------------
            if pk_nulls.sum() == 0 and sales_nulls == 0 and dup_count == 0:
                st.markdown("## Exploratory Data Analysis (EDA)")
                st.markdown("### Pareto Analysis - What drives 80% of your sales?")

                def pareto_table(data, group_col, label):
                    grouped = data.groupby(group_col)['sales'].sum().sort_values(ascending=False).reset_index()
                    grouped['cumulative_pct'] = grouped['sales'].cumsum() / grouped['sales'].sum()
                    threshold_count = (grouped['cumulative_pct'] <= 0.8).sum()
                    summary_text = f"**{threshold_count} {label.lower()}** account for **80% of total sales**."
                    return grouped, summary_text

                df['item_store'] = df['item'].astype(str) + "_" + df['store'].astype(str)

                item_table, item_msg = pareto_table(df, 'item', "Items")
                store_table, store_msg = pareto_table(df, 'store', "Stores")
                combo_table, combo_msg = pareto_table(df, 'item_store', "Item-Store Combinations")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Items")
                    st.write(item_msg)
                    st.dataframe(item_table.head(5), use_container_width=True)
                    st.download_button("Download Items", item_table.to_csv(index=False).encode('utf-8'), "pareto_item.csv")

                with col2:
                    st.subheader("Stores")
                    st.write(store_msg)
                    st.dataframe(store_table.head(5), use_container_width=True)
                    st.download_button("Download Stores", store_table.to_csv(index=False).encode('utf-8'), "pareto_store.csv")

                with col3:
                    st.subheader("Item-Store Combos")
                    st.write(combo_msg)
                    st.dataframe(combo_table.head(5), use_container_width=True)
                    st.download_button("Download Combos", combo_table.to_csv(index=False).encode('utf-8'), "pareto_combo.csv")

                # -----------------------------------
                # 📈 Sales Trend Over Time
                # -----------------------------------
                st.markdown("## Sales Trend Over Time")
                sales_over_time = df.groupby('date')['sales'].sum().reset_index()
                st.line_chart(sales_over_time.set_index('date'))

                # -----------------------------------
                # 📅 Avg Sales by Weekday
                # -----------------------------------
                st.markdown("## Average Sales by Day of the Week")
                df['weekday'] = df['date'].dt.day_name()
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                avg_by_day = df.groupby('weekday')['sales'].mean().reindex(weekday_order)
                st.bar_chart(avg_by_day)

                # -----------------------------------
                # 📆 Monthly Sales Trend
                # -----------------------------------
                st.markdown("## Monthly Sales Trend")
                df['month'] = df['date'].dt.to_period('M').astype(str)
                monthly_sales = df.groupby('month')['sales'].sum().reset_index()
                st.line_chart(monthly_sales.set_index('month'))

                # -----------------------------------
                # 🏆 Top/Bottom Performers
                # -----------------------------------
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Top 5 Stores by Sales")
                    top_stores = df.groupby('store')['sales'].sum().sort_values(ascending=False).head(5).reset_index()
                    st.dataframe(top_stores, use_container_width=True)

                with col2:
                    st.markdown("### Top 5 Items by Sales")
                    top_items = df.groupby('item')['sales'].sum().sort_values(ascending=False).head(5).reset_index()
                    st.dataframe(top_items, use_container_width=True)

                st.markdown("### Bottom 5 Item-Store Combinations")
                bottom_combos = df.groupby('item_store')['sales'].sum().sort_values().head(5).reset_index()
                st.dataframe(bottom_combos, use_container_width=True)

                # -----------------------------------
                # 📦 Sales Distribution Summary
                # -----------------------------------
                st.markdown("## Sales Distribution Summary")
                stats = df['sales'].describe()[['min', '25%', '50%', 'mean', '75%', 'max', 'std']].rename_axis("Metric").reset_index(name="Value")
                st.dataframe(stats, use_container_width=True)

                # -----------------------------------
                # 🧠 Mover & Variation Classification
                # -----------------------------------
                st.markdown("## Mover and Variation Classification")
                df_sorted = df.sort_values(by=['item', 'store', 'date'])

                gap_stats = df_sorted.groupby('item_store')['date'].apply(lambda x: x.diff().dt.days.mean()).reset_index(name='mean_gap_days')
                if gap_stats['mean_gap_days'].nunique() >= 3:
                    gap_stats['mover_type'] = pd.qcut(gap_stats['mean_gap_days'], q=3, labels=['Fast', 'Medium', 'Slow'], duplicates='drop')
                else:
                    gap_stats['mover_type'] = 'Undefined'

                variation_stats = df.groupby('item_store')['sales'].agg(['mean', 'std']).reset_index()
                variation_stats['cv'] = variation_stats['std'] / variation_stats['mean']
                if variation_stats['cv'].nunique() >= 3:
                    variation_stats['variation_type'] = pd.qcut(variation_stats['cv'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                else:
                    variation_stats['variation_type'] = 'Undefined'

                movement_analysis = pd.merge(gap_stats, variation_stats[['item_store', 'cv', 'variation_type']], on='item_store')
                st.markdown("### Item-Store Movement Summary")
                st.dataframe(movement_analysis.sort_values('mean_gap_days').head(10), use_container_width=True)

                st.download_button(
                    "Download Mover & Variation Classification",
                    movement_analysis.to_csv(index=False).encode('utf-8'),
                    "mover_variation_analysis.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Awaiting file upload...")
