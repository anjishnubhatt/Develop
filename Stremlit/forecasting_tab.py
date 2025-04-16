import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import numpy as np
import altair as alt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

warnings.filterwarnings("ignore")


def run():
    st.header("Forecasting with Prophet & SARIMA")

    if "df" not in st.session_state:
        st.warning("Please upload data in the Analysis tab first.")
        return

    df = st.session_state["df"]

    try:
        results = []
        forecast_output = []
        sarima_output = []
        sarima_error_results = []

        for (item, store), group in df.groupby(['item', 'store']):
            group = group.sort_values('date')
            daily_sales = group.groupby('date')['sales'].sum().reset_index()
            daily_sales.columns = ['ds', 'y']

            if len(daily_sales) < 10:
                continue

            # Prophet Forecast
            model = Prophet()
            model.fit(daily_sales)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            fitted = forecast.iloc[:len(daily_sales)].copy()
            fitted['item'] = item
            fitted['store'] = store
            fitted['actual'] = daily_sales['y'].values
            fitted['mape'] = np.abs(fitted['actual'] - fitted['yhat']) / np.maximum(1e-5, np.abs(fitted['actual'])) * 100
            fitted['smape'] = 200 * np.abs(fitted['actual'] - fitted['yhat']) / (np.abs(fitted['actual']) + np.abs(fitted['yhat']) + 1e-5)
            fitted['rmse'] = (fitted['actual'] - fitted['yhat']) ** 2
            fitted['mad'] = np.abs(fitted['actual'] - fitted['yhat'])
            results.append(fitted[['ds', 'item', 'store', 'actual', 'yhat', 'mape', 'smape', 'rmse', 'mad']])
            forecast_output.append(forecast.tail(30).assign(item=item, store=store))

            # SARIMA Forecast
            y = daily_sales.set_index('ds')['y']
            p = d = q = range(0, 2)
            pdq = list(product(p, d, q))
            seasonal_pdq = [(x[0], x[1], x[2], 7) for x in pdq]
            best_aic = float("inf")
            best_order = None
            best_seasonal = None
            for param in pdq:
                for seasonal in seasonal_pdq:
                    try:
                        model = SARIMAX(y, order=param, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
                        results_temp = model.fit(disp=False)
                        if results_temp.aic < best_aic:
                            best_aic = results_temp.aic
                            best_order = param
                            best_seasonal = seasonal
                    except:
                        continue

            model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal)
            sarima_fit = model.fit()
            forecast_sarima = sarima_fit.get_forecast(steps=30)
            forecast_df = forecast_sarima.conf_int()
            forecast_df['yhat'] = forecast_sarima.predicted_mean
            forecast_df['date'] = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=30)
            forecast_df['item'] = item
            forecast_df['store'] = store
            sarima_output.append(forecast_df[['date', 'yhat', 'lower y', 'upper y', 'item', 'store']])

            # SARIMA In-Sample
            fitted_vals = sarima_fit.fittedvalues
            actual_vals = y
            sarima_df = pd.DataFrame({
                'date': actual_vals.index,
                'sales': actual_vals.values,
                'forecast': fitted_vals.values,
                'item': item,
                'store': store
            })
            sarima_df['mape'] = np.abs(sarima_df['sales'] - sarima_df['forecast']) / np.maximum(1e-5, np.abs(sarima_df['sales'])) * 100
            sarima_df['smape'] = 200 * np.abs(sarima_df['sales'] - sarima_df['forecast']) / (np.abs(sarima_df['sales']) + np.abs(sarima_df['forecast']) + 1e-5)
            sarima_df['rmse'] = (sarima_df['sales'] - sarima_df['forecast']) ** 2
            sarima_df['mad'] = np.abs(sarima_df['sales'] - sarima_df['forecast'])
            sarima_error_results.append(sarima_df)

        if not results:
            st.warning("Not enough data to forecast any item-store combination.")
            return

        error_df = pd.concat(results)
        error_df['rmse'] = np.sqrt(error_df['rmse'])
        error_df.rename(columns={"ds": "date", "actual": "sales", "yhat": "forecast"}, inplace=True)

        st.subheader("In-Sample Error Table (Prophet)")
        st.dataframe(error_df.head(), use_container_width=True)
        st.download_button("Download Prophet Error Table", error_df.to_csv(index=False).encode("utf-8"), "in_sample_errors_prophet.csv")

        st.subheader("Aggregated Error Metrics (Prophet)")
        summary_metrics = pd.DataFrame({
            'Metric': ['MAPE', 'sMAPE', 'MAD'],
            'Value': [f"{error_df['mape'].mean():.2f}%", f"{error_df['smape'].mean():.2f}%", f"{error_df['mad'].mean():.2f}"]
        })
        st.dataframe(summary_metrics, use_container_width=True)

        st.subheader("Interactive In-Sample Forecast (Prophet) - First Item-Store")
        preview_df = error_df[(error_df['item'] == error_df['item'].iloc[0]) & (error_df['store'] == error_df['store'].iloc[0])]
        chart = alt.Chart(preview_df).transform_fold(['sales', 'forecast'], as_=['Type', 'Value']).mark_line().encode(
            x='date:T', y='Value:Q', color='Type:N', tooltip=['date:T', 'Type:N', 'Value:Q']
        ).interactive().properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        st.subheader("30-Day Forecast (Prophet) - First Item-Store")
        forecast_table = forecast_output[0][['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'item', 'store']]
        forecast_table.rename(columns={'ds': 'date', 'yhat': 'forecast'}, inplace=True)
        st.dataframe(forecast_table, use_container_width=True)
        st.download_button("Download 30-Day Forecast (Prophet)", forecast_table.to_csv(index=False).encode("utf-8"), "forecast_30_days_prophet.csv")

        # SARIMA error table
        sarima_errors = pd.concat(sarima_error_results)
        sarima_errors['rmse'] = np.sqrt(sarima_errors['rmse'])

        st.subheader("In-Sample Error Table (SARIMA)")
        st.dataframe(sarima_errors.head(), use_container_width=True)
        st.download_button("Download SARIMA Error Table", sarima_errors.to_csv(index=False).encode("utf-8"), "in_sample_errors_sarima.csv")

        st.subheader("Aggregated Error Metrics (SARIMA)")
        sarima_summary = pd.DataFrame({
            'Metric': ['MAPE', 'sMAPE', 'MAD'],
            'Value': [f"{sarima_errors['mape'].mean():.2f}%", f"{sarima_errors['smape'].mean():.2f}%", f"{sarima_errors['mad'].mean():.2f}"]
        })
        st.dataframe(sarima_summary, use_container_width=True)

        st.subheader("Interactive In-Sample Forecast (SARIMA) - First Item-Store")
        sarima_preview = sarima_errors[(sarima_errors['item'] == sarima_errors['item'].iloc[0]) & (sarima_errors['store'] == sarima_errors['store'].iloc[0])]

        chart2 = alt.Chart(sarima_preview).transform_fold(['sales', 'forecast'], as_=['Type', 'Value']).mark_line().encode(
            x='date:T', y='Value:Q', color='Type:N', tooltip=['date:T', 'Type:N', 'Value:Q']
        ).interactive().properties(height=300)
        st.altair_chart(chart2, use_container_width=True)

        st.subheader("30-Day Forecast (SARIMA) - First Item-Store")
        sarima_table = sarima_output[0].rename(columns={'yhat': 'forecast', 'lower y': 'yhat_lower', 'upper y': 'yhat_upper'})
        st.dataframe(sarima_table, use_container_width=True)
        st.download_button("Download 30-Day Forecast (SARIMA)", sarima_table.to_csv(index=False).encode("utf-8"), "forecast_30_days_sarima.csv")


        # ----------------------------
        # 🔀 Ensemble: Best Model Per Item-Store
        # ----------------------------
        st.subheader("🔀 Ensemble: Best Model Selected Per Item-Store")

        # Tag source model in both error sets
        error_df['model'] = 'Prophet'
        sarima_errors['model'] = 'SARIMA'

        # Combine all
        all_errors = pd.concat([error_df, sarima_errors.rename(columns={"date": "ds"})], ignore_index=True)

        # Select best model by MAPE
        
        best_models = pd.merge(all_errors.groupby(['item', 'store', 'model'])['mape'].mean().reset_index().sort_values('mape').drop_duplicates(['item', 'store'])[['item', 'store', 'model']], all_errors, on=['item', 'store', 'model'], how='left')

        best_models['rmse'] = np.sqrt(best_models['rmse'])

        st.markdown("### In-Sample Error Table (Best Models)")
        st.dataframe(best_models[['ds', 'item', 'store', 'sales', 'forecast', 'model', 'mape', 'smape', 'mad']], use_container_width=True)
        st.download_button("📥 Download Ensemble Error Table", best_models.to_csv(index=False).encode('utf-8'), "ensemble_in_sample.csv")

        # Aggregated metrics
        st.markdown("### Aggregated Error Metrics (Best Models)")
        agg = pd.DataFrame({
            'Metric': ['MAPE', 'sMAPE', 'MAD'],
            'Value': [f"{best_models['mape'].mean():.2f}%", f"{best_models['smape'].mean():.2f}%", f"{best_models['mad'].mean():.2f}"]
        })
        st.dataframe(agg, use_container_width=True)

        # Interactive chart (first item-store)
        st.markdown("### In-Sample Forecast Chart (Best Model)")
        sample = best_models[(best_models['item'] == best_models['item'].iloc[0]) & (best_models['store'] == best_models['store'].iloc[0])]
        chart_ensemble = alt.Chart(sample).transform_fold(
            ['sales', 'forecast'], as_=['Type', 'Value']
        ).mark_line().encode(
            x='ds:T', y='Value:Q', color='Type:N', tooltip=['ds:T', 'Type:N', 'Value:Q']
        ).interactive().properties(height=300)
        st.altair_chart(chart_ensemble, use_container_width=True)

        # Forecast output: choose model-specific forecast
        st.markdown("### 30-Day Forecast Table (Best Models)")

        # Tag model in forecast sets
        prophet_fc = pd.concat(forecast_output).rename(columns={'ds': 'date', 'yhat': 'forecast'})
        prophet_fc['model'] = 'Prophet'
        sarima_fc = pd.concat(sarima_output).rename(columns={'yhat': 'forecast', 'lower y': 'yhat_lower', 'upper y': 'yhat_upper'})
        sarima_fc['model'] = 'SARIMA'

        all_forecasts = pd.concat([prophet_fc, sarima_fc], ignore_index=True)

        # Filter only forecasts from selected models
        best_forecasts = pd.merge(
            best_models[['item', 'store', 'model']],
            all_forecasts,
            on=['item', 'store', 'model'],
            how='left'
        )
        best_forecasts=best_forecasts[['date','item','store','model','forecast','yhat_upper','yhat_lower']]


        st.dataframe(best_forecasts.head(), use_container_width=True)
        st.download_button("📥 Download Ensemble 30-Day Forecast", best_forecasts.to_csv(index=False).encode('utf-8'), "ensemble_forecast.csv")


    except Exception as e:
        st.error(f"Forecasting error: {e}")