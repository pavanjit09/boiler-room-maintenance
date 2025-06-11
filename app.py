import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Helper functions ---
def load_and_preprocess_data(filepath):
    """Load and preprocess the sensor data"""
    df = pd.read_csv(filepath, parse_dates=['Timestamp'])
    
    features = ['Timestamp', 'Main steam flow (t/h)', 'Main steam temperature (boiler side) (℃)',
               'Main steam pressure (boiler side) (Mpa)', 'Feedwater temperature (℃)',
               'Feedwater flow (t/h)', 'Flue gas temperature (℃)', 'Boiler oxygen level (%)',
               'CO (mg/m3)', 'CO2 (ppm)', 'Opacity (%)', 'Dust (mg/m3)', 'Boiler Eff (%)',
               'Coal Flow (t/h)', 'Gross Load (MW)', 'HHV (Kcal/Kg)']
    
    df = df[features].interpolate().drop_duplicates(subset=['Timestamp']).set_index('Timestamp')
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), 
                           columns=df.columns, 
                           index=df.index)
    return df, df_scaled, scaler

def create_sequences(data, window_size=30):
    """Create sequences for time series analysis"""
    seq = []
    for i in range(len(data) - window_size + 1):
        seq.append(data[i:i+window_size])
    return np.array(seq)

def detect_anomalies(df_scaled, model, threshold=0.01):
    """Detect anomalies using reconstruction error"""
    X = create_sequences(df_scaled.values)
    preds = model.predict(X)
    
    # Calculate reconstruction error
    losses = np.mean(np.abs(preds - X), axis=(1,2))
    padded_losses = np.concatenate([np.zeros(29), losses])  # 30-1=29
    
    # Create results dataframe
    result = df_scaled.copy()
    result['Reconstruction_Error'] = padded_losses
    result['Threshold'] = threshold
    result['Anomaly'] = result['Reconstruction_Error'] > threshold
    return result

def predict_anomalies(df_scaled, model, threshold=0.01):
    """Predict future anomalies (fixed 5-step forecast)"""
    forecast_steps = 5
    window_size = 30
    
    if len(df_scaled) < (window_size + forecast_steps):
        raise ValueError(f"Need at least {window_size + forecast_steps} data points")
    
    X, idx = [], []
    y_true = []

    for i in range(len(df_scaled) - window_size - forecast_steps + 1):
        X.append(df_scaled.iloc[i:i+window_size].values)
        y_true.append(df_scaled.iloc[i+window_size:i+window_size+forecast_steps].values)
        idx.append(df_scaled.index[i+window_size])

    X = np.array(X)
    preds = model.predict(X)
    
    # Reshape predictions (samples, forecast_steps, features)
    preds_reshaped = preds.reshape(-1, forecast_steps, df_scaled.shape[1])
    errors = np.mean(np.abs(preds_reshaped - np.array(y_true)), axis=(1, 2))
    
    # Create output dataframe (Confidence removed)
    df_out = pd.DataFrame({
        'Timestamp': idx,
        'Forecast_Error': errors,
        'Threshold': threshold,
        'Anomaly_Flag': errors > threshold
    }).set_index('Timestamp')

    # Add forecasted values
    for step in range(forecast_steps):
        for col_i, col in enumerate(df_scaled.columns):
            df_out[f'Step_{step+1}_{col}'] = preds_reshaped[:, step, col_i]

    return df_out

# --- Load Models ---
try:
    det_model = load_model(os.path.join('models', 'anomaly_detection_model.keras'))
    pred_model = load_model(os.path.join('models', 'anomaly_forecast_model.keras'))
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# --- Streamlit App ---
st.title("Boiler Sensor Anomaly Analysis")

page = st.sidebar.selectbox("Choose your tool:", ["Anomaly Detection", "Anomaly Prediction"])

uploaded = st.sidebar.file_uploader("📁 Upload sensor CSV", type="csv")
if not uploaded:
    st.info("𝐔𝐩𝐥𝐨𝐚𝐝 𝐂𝐒𝐕 𝐟𝐢𝐥𝐞 𝐭𝐨 𝐏𝐫𝐨𝐜𝐞𝐞𝐝")
    st.info("""
This application provides intelligent monitoring and analysis of **boiler sensor data** to enhance operational safety and efficiency.

### 🔍 Features:
- **Anomaly Detection:** Identifies unexpected behavior in historical sensor data using a trained deep learning model. Useful for reviewing past performance and detecting operational issues.
- **Anomaly Prediction:** Forecasts future anomalies based on recent data trends. Enables proactive action before faults occur, helping prevent downtime or equipment damage.

### 📈 How it works:
- The models analyze key boiler parameters like steam pressure, feedwater flow, temperature, emissions, and load.
- Anomalies are identified using reconstruction or forecast error — when the actual system behavior significantly deviates from the model’s learned patterns.

This tool is designed for **plant operators, engineers, and analysts** seeking to optimize boiler health, improve maintenance planning, and reduce operational risks.
""")
    st.stop()

try:
    df_raw, df_scaled, scaler = load_and_preprocess_data(uploaded)
except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.stop()

if page == "Anomaly Detection":
    st.header("Anomaly Detection")
    
    # Feature selection with "All" option
    feature_options = ["All Features"] + list(df_scaled.columns) + ["Reconstruction Error"]
    feature = st.selectbox("Choose feature to display:", feature_options)
    
    threshold = st.number_input("Anomaly Threshold", value=0.01, step=0.001, min_value=0.0)
    
    if st.button("Detect Anomalies"):
        with st.spinner("Analyzing data..."):
            result = detect_anomalies(df_scaled, det_model, threshold=threshold)
            
            # Plot reconstruction error with dynamic threshold
            fig = px.line(result, x=result.index, y='Reconstruction_Error', 
                         title="Reconstruction Error Over Time")
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"Threshold: {threshold:.3f}")
            fig.add_scatter(
                x=result.index[result['Anomaly']], 
                y=result['Reconstruction_Error'][result['Anomaly']], 
                mode='markers', 
                marker_color='red',
                name="Anomaly"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomalies table with selected features
            anomalies = result[result['Anomaly']].copy()
            anomalies['Timestamp'] = anomalies.index
            
            if feature == "All Features":
                display_cols = df_scaled.columns.tolist() + ['Reconstruction_Error', 'Threshold']
            elif feature == "Reconstruction Error":
                display_cols = ['Reconstruction_Error', 'Threshold']
            else:
                display_cols = [feature, 'Reconstruction_Error', 'Threshold']
            
            st.dataframe(anomalies[display_cols])
            st.download_button(
                "⬇️ Download Anomalies CSV", 
                anomalies[display_cols].to_csv(), 
                file_name="anomalies.csv"
            )

elif page == "Anomaly Prediction":
    st.header("Anomaly Prediction")
    st.info("Note: Model predicts fixed 5 steps ahead")

    threshold = st.number_input("Prediction Threshold", value=0.01, step=0.001, min_value=0.0)
    
    if st.button("Predict Future Anomalies"):
        with st.spinner("Making predictions..."):
            try:
                pred_df = predict_anomalies(df_scaled, pred_model, threshold=threshold)
                st.dataframe(pred_df)
                st.download_button(
                    "⬇️ Download Forecast CSV", 
                    pred_df.to_csv(), 
                    file_name="forecast.csv"
                )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

st.success("✅ Analysis complete!")

