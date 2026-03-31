"""
Streamlit UI for ML-IDS
Interactive interface for network flow classification
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

# Try to import TensorFlow (optional for DL models)
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, FEATURE_COLUMNS


# Page configuration
st.set_page_config(
    page_title="ML-IDS: Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        # ML Supervised Models
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'Histogram Gradient Boosting': 'histogram_gradient_boosting.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Linear SVM': 'linear_svm.pkl',
        'Extra Trees': 'extra_trees.pkl',
        'Balanced Random Forest': 'balanced_random_forest.pkl',
        
        # ML Anomaly Detection Models
        'Isolation Forest': 'isolation_forest.pkl',
        'Elliptic Envelope': 'elliptic_envelope.pkl',
        'PCA Reconstruction': 'pca_reconstruction.pkl',
        'kNN Anomaly Detection': 'knn_anomaly_detection.pkl',
        'HBOS': 'hbos.pkl',
        'COPOD': 'copod.pkl',
        
        # DL Supervised Models (only if TensorFlow available)
        'MLP': 'mlp_model.h5',
        'Transformer': 'transformer_model.h5',
        'Vanilla LSTM': 'vanilla_lstm_model.h5',
        'Stacked LSTM': 'stacked_lstm_model.h5',
        'Bidirectional LSTM': 'bidirectional_lstm_model.h5',
        
        # DL Anomaly Detection Models (only if TensorFlow available)
        'AutoEncoder': 'autoencoder_model.h5',
        'LSTM AutoEncoder': 'lstm_autoencoder_model.h5',
        'Denoising AutoEncoder': 'denoising_autoencoder_model.h5'
    }
    
    for name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            try:
                if filename.endswith('.h5'):
                    # Skip DL models if TensorFlow not available
                    if not TENSORFLOW_AVAILABLE:
                        continue
                    # Load with custom objects to handle serialization issues
                    models[name] = keras.models.load_model(
                        model_path,
                        custom_objects={'mse': keras.losses.MeanSquaredError()},
                        compile=False
                    )
                else:
                    models[name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load {name}: {str(e)}")
    
    # Try to load TabNet (only if PyTorch available)
    tabnet_path = os.path.join(MODELS_DIR, 'tabnet_model.zip')
    if os.path.exists(tabnet_path):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            tabnet = TabNetClassifier()
            tabnet.load_model(tabnet_path.replace('.zip', ''))
            models['TabNet'] = tabnet
        except Exception as e:
            pass  # Silently skip if PyTorch not available
    
    return models


@st.cache_resource
def load_scaler():
    """Load the feature scaler"""
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    
    # Fallback to old location
    scaler_path_old = os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path_old):
        return joblib.load(scaler_path_old)
    
    return None


def preprocess_input(df, scaler):
    """
    Preprocess input data - extract universal features and scale
    Works with any dataset that has the required columns
    Handles different column name variations
    """
    # Column name mapping - handles different naming conventions
    # CICIDS2017 uses "Total Fwd Packets", CICIDS2018 uses "Tot Fwd Pkts"
    column_mappings = {
        'Flow Duration': ['Flow Duration', 'flow_duration', 'duration'],
        'Destination Port': ['Destination Port', 'Dst Port', 'dst_port'],
        'Total Fwd Packets': ['Total Fwd Packets', 'Tot Fwd Pkts', 'tot_fwd_pkts'],
        'Total Backward Packets': ['Total Backward Packets', 'Tot Bwd Pkts', 'tot_bwd_pkts'],
        'Flow Packets/s': ['Flow Packets/s', 'Flow Pkts/s', 'flow_pkts_s'],
        'Flow Bytes/s': ['Flow Bytes/s', 'Flow Byts/s', 'flow_byts_s'],
        'Fwd Packet Length Mean': ['Fwd Packet Length Mean', 'Fwd Pkt Len Mean', 'fwd_pkt_len_mean'],
        'Fwd Packet Length Max': ['Fwd Packet Length Max', 'Fwd Pkt Len Max', 'fwd_pkt_len_max'],
        'Bwd Packet Length Mean': ['Bwd Packet Length Mean', 'Bwd Pkt Len Mean', 'bwd_pkt_len_mean'],
        'Packet Length Mean': ['Packet Length Mean', 'Pkt Len Mean', 'pkt_len_mean'],
        'Packet Length Std': ['Packet Length Std', 'Pkt Len Std', 'pkt_len_std'],
        'Flow IAT Mean': ['Flow IAT Mean', 'flow_iat_mean'],
        'Flow IAT Std': ['Flow IAT Std', 'flow_iat_std'],
        'Flow IAT Max': ['Flow IAT Max', 'flow_iat_max'],
        'SYN Flag Count': ['SYN Flag Count', 'SYN Flag Cnt', 'syn_flag_cnt'],
        'ACK Flag Count': ['ACK Flag Count', 'ACK Flag Cnt', 'ack_flag_cnt'],
        'RST Flag Count': ['RST Flag Count', 'RST Flag Cnt', 'rst_flag_cnt'],
        'Average Packet Size': ['Average Packet Size', 'Pkt Size Avg', 'pkt_size_avg']
    }
    
    # Map columns
    df_features = pd.DataFrame()
    available_features = []
    missing_features = []
    
    for target_col in FEATURE_COLUMNS:
        found = False
        possible_names = column_mappings.get(target_col, [target_col])
        
        for possible_name in possible_names:
            if possible_name in df.columns:
                df_features[target_col] = pd.to_numeric(df[possible_name], errors='coerce')
                available_features.append(target_col)
                found = True
                break
        
        if not found:
            df_features[target_col] = 0
            missing_features.append(target_col)
    
    # Show summary
    if missing_features:
        st.warning(f"⚠️ {len(missing_features)} features missing - will use 0 for: {', '.join(missing_features[:3])}{'...' if len(missing_features) > 3 else ''}")
    
    # Handle infinite and missing values
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.fillna(0, inplace=True)
    
    # Scale features using the saved scaler
    if scaler:
        df_scaled = scaler.transform(df_features)
        st.info(f"✅ Extracted {len(available_features)}/{len(FEATURE_COLUMNS)} features and applied StandardScaler")
    else:
        df_scaled = df_features.values
        st.warning("⚠️ No scaler found - using raw features")
    
    return df_scaled


def predict_all_models(models, data):
    """
    Make predictions using all models
    
    Returns:
        Dictionary with model names as keys and (predictions, confidence) as values
    """
    results = {}
    
    for model_name, model in models.items():
        try:
            # AutoEncoder models (reconstruction-based)
            if 'AutoEncoder' in model_name or 'Autoencoder' in model_name:
                # Predict reconstruction
                reconstructed = model.predict(data, verbose=0)
                # Calculate reconstruction error
                mse = np.mean(np.square(data - reconstructed), axis=1)
                # Threshold: top 10% errors are anomalies
                threshold = np.percentile(mse, 90)
                prediction = (mse > threshold).astype(int)
                # Normalize confidence to 0-1
                confidence = np.clip(mse / (threshold * 2), 0, 1)
            
            # LSTM supervised models (but NOT AutoEncoders)
            elif 'LSTM' in model_name:
                # Reshape for LSTM
                data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
                prediction_proba = model.predict(data_reshaped, verbose=0)
                prediction = (prediction_proba > 0.5).astype(int).flatten()
                confidence = prediction_proba.flatten()
            
            # MLP and Transformer models
            elif 'MLP' in model_name or 'Transformer' in model_name:
                prediction_proba = model.predict(data, verbose=0)
                prediction = (prediction_proba > 0.5).astype(int).flatten()
                confidence = prediction_proba.flatten()
            
            # TabNet
            elif model_name == 'TabNet':
                prediction = model.predict(data)
                confidence = prediction.astype(float)
            
            # ML models
            else:
                prediction = model.predict(data)
                
                # Handle anomaly detection models that return -1/1
                if set(np.unique(prediction)).issubset({-1, 1}):
                    prediction = np.where(prediction == -1, 1, 0)
                
                # Get confidence if available
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(data)
                    confidence = prediction_proba[:, 1]
                else:
                    confidence = prediction.astype(float)
            
            results[model_name] = (prediction, confidence)
            
        except Exception as e:
            # Skip models that fail
            print(f"Skipped {model_name}: {str(e)}")
            pass
    
    return results


def predict_flow(models, data, model_name):
    """Make prediction using selected model"""
    model = models[model_name]
    
    if model_name == 'Neural Network':
        prediction_proba = model.predict(data, verbose=0)
        prediction = (prediction_proba > 0.5).astype(int).flatten()
        confidence = prediction_proba.flatten()
    else:
        prediction = model.predict(data)
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(data)
            confidence = prediction_proba[:, 1]
        else:
            confidence = prediction
    
    return prediction, confidence


def main():
    # Header
    st.title("🛡️ ML-IDS: Machine Learning Intrusion Detection System")
    st.markdown("### Classify network flows as **Benign** or **Malicious**")
    
    # Show warning if TensorFlow not available
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not available. Deep Learning models will be skipped. Only ML models will be loaded.")
        st.info("💡 To use DL models, install Python 3.11 and run: `pip install tensorflow`")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Load models and scaler
    with st.spinner("Loading models..."):
        models = load_models()
        scaler = load_scaler()
    
    if not models:
        st.error("❌ No trained models found! Please train models first.")
        st.info("Run: `python src/models/train_model.py`")
        return
    
    st.sidebar.success(f"✅ Loaded {len(models)} models")
    
    # Show loaded models
    with st.sidebar.expander("📋 Available Models"):
        for model_name in models.keys():
            st.write(f"• {model_name}")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📁 Upload Dataset", "🤖 AI Generate Flows"])
    
    # Tab 1: Upload Dataset
    with tab1:
        st.header("Upload Network Flow Dataset")
        st.markdown("Upload any CSV file with network flow data. The app will automatically extract the 18 universal features.")
        
        # API Key for LLM analysis
        groq_api_key_upload = st.text_input("Groq API Key (for AI analysis)", type="password", 
                                            help="Optional: Get your free API key from https://console.groq.com for detailed AI insights",
                                            key="groq_upload")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(df)} flows detected.")
                
                # Show available columns for debugging
                with st.expander("📋 Available Columns in Uploaded File"):
                    st.write(f"Total columns: {len(df.columns)}")
                    st.write(list(df.columns))
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10))
                
                # Preprocess button
                if st.button("🔍 Analyze Dataset", type="primary"):
                    # Preprocess
                    df_processed = preprocess_input(df.copy(), scaler)
                    
                    st.success("✅ Data preprocessed! Running all models...")
                    
                    # Create placeholder for results
                    results_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run models one by one and show results progressively
                    all_results = {}
                    comparison_data = []
                    total_models = len(models)
                    
                    for idx, (model_name, model) in enumerate(models.items()):
                        status_text.text(f"Running {model_name}... ({idx+1}/{total_models})")
                        
                        try:
                            # Predict with single model
                            # Check AutoEncoder FIRST (before LSTM check)
                            if 'AutoEncoder' in model_name or 'Autoencoder' in model_name:
                                reconstructed = model.predict(df_processed, verbose=0)
                                mse = np.mean(np.square(df_processed - reconstructed), axis=1)
                                threshold = np.percentile(mse, 90)
                                predictions = (mse > threshold).astype(int)
                                confidence = np.clip(mse / (threshold * 2), 0, 1)
                            
                            # LSTM supervised models (but NOT AutoEncoders)
                            elif 'LSTM' in model_name:
                                data_reshaped = df_processed.reshape((df_processed.shape[0], 1, df_processed.shape[1]))
                                prediction_proba = model.predict(data_reshaped, verbose=0)
                                predictions = (prediction_proba > 0.5).astype(int).flatten()
                                confidence = prediction_proba.flatten()
                            
                            elif 'MLP' in model_name or 'Transformer' in model_name:
                                prediction_proba = model.predict(df_processed, verbose=0)
                                predictions = (prediction_proba > 0.5).astype(int).flatten()
                                confidence = prediction_proba.flatten()
                            
                            elif model_name == 'TabNet':
                                predictions = model.predict(df_processed)
                                confidence = predictions.astype(float)
                            
                            else:
                                predictions = model.predict(df_processed)
                                if set(np.unique(predictions)).issubset({-1, 1}):
                                    predictions = np.where(predictions == -1, 1, 0)
                                
                                if hasattr(model, 'predict_proba'):
                                    prediction_proba = model.predict_proba(df_processed)
                                    confidence = prediction_proba[:, 1]
                                else:
                                    confidence = predictions.astype(float)
                            
                            # Store results
                            all_results[model_name] = (predictions, confidence)
                            
                            # Add to comparison
                            benign_count = sum(predictions == 0)
                            malicious_count = sum(predictions == 1)
                            avg_conf = np.mean(confidence)
                            
                            comparison_data.append({
                                'Model': model_name,
                                'Benign': benign_count,
                                'Malicious': malicious_count,
                                'Benign %': f"{benign_count/len(df)*100:.1f}%",
                                'Malicious %': f"{malicious_count/len(df)*100:.1f}%",
                                'Avg Confidence': f"{avg_conf:.2%}"
                            })
                            
                            # Update display progressively
                            with results_placeholder.container():
                                st.markdown(f"### 📊 Results ({len(comparison_data)}/{total_models} models)")
                                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                        
                        except Exception as e:
                            status_text.text(f"⚠️ Skipped {model_name}: {str(e)}")
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / total_models)
                    
                    status_text.text("✅ All models completed!")
                    progress_bar.empty()
                    
                    # Final comparison
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    st.markdown("### 🏆 Final Results - All Models")
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    # Visualization
                    fig = go.Figure()
                    for _, row in df_comparison.iterrows():
                        fig.add_trace(go.Bar(
                            name=row['Model'],
                            x=['Benign', 'Malicious'],
                            y=[row['Benign'], row['Malicious']],
                            text=[row['Benign'], row['Malicious']],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="Classification Results - All Models",
                        xaxis_title="Classification",
                        yaxis_title="Count",
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence distribution
                        fig_conf = go.Figure()
                        for model_name, (predictions, confidence) in list(all_results.items())[:5]:
                            fig_conf.add_trace(go.Box(
                                y=confidence,
                                name=model_name,
                                boxmean='sd'
                            ))
                        fig_conf.update_layout(
                            title="Confidence Distribution (Top 5 Models)",
                            yaxis_title="Confidence Score",
                            height=400
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    with col2:
                        # Model agreement
                        agreement_scores = []
                        model_names = list(all_results.keys())
                        for i, name in enumerate(model_names):
                            preds = all_results[name][0]
                            agreements = [np.mean(preds == all_results[other][0]) * 100 
                                        for other in model_names if other != name]
                            if agreements:
                                agreement_scores.append({
                                    'Model': name,
                                    'Avg Agreement': np.mean(agreements)
                                })
                        
                        df_agreement = pd.DataFrame(agreement_scores).sort_values('Avg Agreement', ascending=False).head(10)
                        fig_agree = px.bar(df_agreement, x='Model', y='Avg Agreement',
                                          title="Model Agreement Score (Top 10)",
                                          labels={'Avg Agreement': 'Agreement %'})
                        fig_agree.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_agree, use_container_width=True)
                    
                    # Generate detailed LLM analysis
                    st.markdown("---")
                    st.markdown("### 🤖 AI-Powered Detailed Analysis")
                    
                    if groq_api_key_upload:
                        with st.spinner("🤖 Analyzing results and generating insights..."):
                            try:
                                from src.report_generator import ReportGenerator
                                
                                report_gen = ReportGenerator(groq_api_key_upload)
                                
                                # Check if we have ground truth
                                has_gt = 'True_Label' in df.columns or 'Label' in df.columns
                                
                                report = report_gen.generate_detailed_report(
                                    df, all_results, has_ground_truth=has_gt
                                )
                                
                                st.markdown(report)
                            except Exception as e:
                                st.warning(f"Could not generate AI analysis: {str(e)}")
                    else:
                        st.info("💡 Enter your Groq API key above to get detailed AI-powered insights about model performance and dataset characteristics")
                    
                    # Download results
                    st.markdown("---")
                    df_result = df.copy()
                    
                    # Add all model predictions
                    for model_name, (predictions, confidence) in all_results.items():
                        df_result[f'{model_name}_Prediction'] = ['Malicious' if p == 1 else 'Benign' for p in predictions]
                        df_result[f'{model_name}_Confidence'] = confidence
                    
                    csv = df_result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results with All Models",
                        data=csv,
                        file_name="ids_results_all_models.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Tab 2: AI Generate Flows
    with tab2:
        st.header("🤖 AI-Generated Network Flows")
        st.markdown("Use AI to generate synthetic network flow data for testing")
        
        # Initialize session state
        if 'generated_flows' not in st.session_state:
            st.session_state.generated_flows = None
        
        # API Key input
        groq_api_key = st.text_input("Groq API Key", type="password", 
                                     help="Get your free API key from https://console.groq.com")
        
        if groq_api_key:
            col1, col2 = st.columns(2)
            
            with col1:
                scenario_type = st.selectbox(
                    "Scenario Type",
                    ["mixed", "benign", "ddos", "port_scan", "brute_force"],
                    help="Type of traffic to generate"
                )
            
            with col2:
                flow_count = st.number_input("Number of Flows", min_value=10, max_value=500, value=100)
            
            if st.button("🎲 Generate Flows", type="primary"):
                try:
                    with st.spinner("🤖 Generating network flows..."):
                        from src.flow_generator import FlowGenerator
                        
                        generator = FlowGenerator(groq_api_key)
                        df_generated = generator.generate_scenario(scenario_type, flow_count)
                        
                        # Store in session state
                        st.session_state.generated_flows = df_generated
                        
                        st.success(f"✅ Generated {len(df_generated)} flows!")
                        
                except Exception as e:
                    st.error(f"❌ Error generating flows: {str(e)}")
            
            # Show generated data if available
            if st.session_state.generated_flows is not None:
                df_generated = st.session_state.generated_flows
                
                # Show original distribution if True_Label exists
                if 'True_Label' in df_generated.columns:
                    st.markdown("### 📊 Original Generated Distribution")
                    true_benign = sum(df_generated['True_Label'] == 0)
                    true_attack = sum(df_generated['True_Label'] == 1)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Flows", len(df_generated))
                    with col2:
                        st.metric("Benign (Ground Truth)", f"{true_benign} ({true_benign/len(df_generated)*100:.1f}%)")
                    with col3:
                        st.metric("Attack (Ground Truth)", f"{true_attack} ({true_attack/len(df_generated)*100:.1f}%)")
                
                # Show preview
                with st.expander("📋 Generated Data Preview"):
                    st.dataframe(df_generated.head(10))
                
                # Analyze button
                if st.button("🔍 Analyze Generated Flows", type="secondary"):
                    # Preprocess
                    df_processed = preprocess_input(df_generated.copy(), scaler)
                    
                    st.success("✅ Data preprocessed! Running all models...")
                    
                    # Create placeholder for results
                    results_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run models one by one and show results progressively
                    all_results = {}
                    comparison_data = []
                    total_models = len(models)
                    
                    for idx, (model_name, model) in enumerate(models.items()):
                        status_text.text(f"Running {model_name}... ({idx+1}/{total_models})")
                        
                        try:
                            # Predict with single model
                            # Check AutoEncoder FIRST (before LSTM check)
                            if 'AutoEncoder' in model_name or 'Autoencoder' in model_name:
                                reconstructed = model.predict(df_processed, verbose=0)
                                mse = np.mean(np.square(df_processed - reconstructed), axis=1)
                                threshold = np.percentile(mse, 90)
                                predictions = (mse > threshold).astype(int)
                                confidence = np.clip(mse / (threshold * 2), 0, 1)
                            
                            # LSTM supervised models (but NOT AutoEncoders)
                            elif 'LSTM' in model_name:
                                data_reshaped = df_processed.reshape((df_processed.shape[0], 1, df_processed.shape[1]))
                                prediction_proba = model.predict(data_reshaped, verbose=0)
                                predictions = (prediction_proba > 0.5).astype(int).flatten()
                                confidence = prediction_proba.flatten()
                            
                            elif 'MLP' in model_name or 'Transformer' in model_name:
                                prediction_proba = model.predict(df_processed, verbose=0)
                                predictions = (prediction_proba > 0.5).astype(int).flatten()
                                confidence = prediction_proba.flatten()
                            
                            elif model_name == 'TabNet':
                                predictions = model.predict(df_processed)
                                confidence = predictions.astype(float)
                            
                            else:
                                predictions = model.predict(df_processed)
                                if set(np.unique(predictions)).issubset({-1, 1}):
                                    predictions = np.where(predictions == -1, 1, 0)
                                
                                if hasattr(model, 'predict_proba'):
                                    prediction_proba = model.predict_proba(df_processed)
                                    confidence = prediction_proba[:, 1]
                                else:
                                    confidence = predictions.astype(float)
                            
                            # Store results
                            all_results[model_name] = (predictions, confidence)
                            
                            # Add to comparison
                            benign_count = sum(predictions == 0)
                            malicious_count = sum(predictions == 1)
                            avg_conf = np.mean(confidence)
                            
                            comparison_data.append({
                                'Model': model_name,
                                'Benign': benign_count,
                                'Malicious': malicious_count,
                                'Benign %': f"{benign_count/len(df_generated)*100:.1f}%",
                                'Malicious %': f"{malicious_count/len(df_generated)*100:.1f}%",
                                'Avg Confidence': f"{avg_conf:.2%}"
                            })
                            
                            # Update display progressively
                            with results_placeholder.container():
                                st.markdown(f"### 📊 Results ({len(comparison_data)}/{total_models} models)")
                                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                        
                        except Exception as e:
                            status_text.text(f"⚠️ Skipped {model_name}: {str(e)}")
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / total_models)
                    
                    status_text.text("✅ All models completed!")
                    progress_bar.empty()
                    
                    # Final comparison
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Add accuracy metrics if True_Label exists
                    if 'True_Label' in df_generated.columns:
                        st.markdown("### 🎯 Model Accuracy vs Ground Truth")
                        
                        true_labels = df_generated['True_Label'].values
                        accuracy_data = []
                        
                        for model_name, (predictions, confidence) in all_results.items():
                            accuracy = np.mean(predictions == true_labels) * 100
                            accuracy_data.append({
                                'Model': model_name,
                                'Accuracy': f"{accuracy:.2f}%"
                            })
                        
                        df_accuracy = pd.DataFrame(accuracy_data)
                        st.dataframe(df_accuracy, use_container_width=True)
                    
                    st.markdown("### 🏆 Final Results - All Models")
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    # Visualization
                    fig = go.Figure()
                    for _, row in df_comparison.iterrows():
                        fig.add_trace(go.Bar(
                            name=row['Model'],
                            x=['Benign', 'Malicious'],
                            y=[row['Benign'], row['Malicious']],
                            text=[row['Benign'], row['Malicious']],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="Classification Results - All Models",
                        xaxis_title="Classification",
                        yaxis_title="Count",
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence distribution
                        fig_conf = go.Figure()
                        for model_name, (predictions, confidence) in list(all_results.items())[:5]:
                            fig_conf.add_trace(go.Box(
                                y=confidence,
                                name=model_name,
                                boxmean='sd'
                            ))
                        fig_conf.update_layout(
                            title="Confidence Distribution (Top 5 Models)",
                            yaxis_title="Confidence Score",
                            height=400
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    with col2:
                        # Model agreement
                        agreement_scores = []
                        model_names = list(all_results.keys())
                        for i, name in enumerate(model_names):
                            preds = all_results[name][0]
                            agreements = [np.mean(preds == all_results[other][0]) * 100 
                                        for other in model_names if other != name]
                            if agreements:
                                agreement_scores.append({
                                    'Model': name,
                                    'Avg Agreement': np.mean(agreements)
                                })
                        
                        df_agreement = pd.DataFrame(agreement_scores).sort_values('Avg Agreement', ascending=False).head(10)
                        fig_agree = px.bar(df_agreement, x='Model', y='Avg Agreement',
                                          title="Model Agreement Score (Top 10)",
                                          labels={'Avg Agreement': 'Agreement %'})
                        fig_agree.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_agree, use_container_width=True)
                    
                    # Generate detailed LLM analysis
                    st.markdown("---")
                    st.markdown("### 🤖 AI-Powered Detailed Analysis")
                    
                    with st.spinner("🤖 Analyzing results and generating insights..."):
                        try:
                            from src.report_generator import ReportGenerator
                            
                            report_gen = ReportGenerator(groq_api_key)
                            report = report_gen.generate_detailed_report(
                                df_generated, all_results, has_ground_truth=True
                            )
                            
                            st.markdown(report)
                        except Exception as e:
                            st.warning(f"Could not generate AI analysis: {str(e)}")
                    
                    # Download
                    df_result = df_generated.copy()
                    
                    # Add all model predictions
                    for model_name, (predictions, confidence) in all_results.items():
                        df_result[f'{model_name}_Prediction'] = ['Malicious' if p == 1 else 'Benign' for p in predictions]
                        df_result[f'{model_name}_Confidence'] = confidence
                    
                    csv = df_result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results with All Models",
                        data=csv,
                        file_name="ai_generated_flows_results.csv",
                        mime="text/csv"
                    )
        else:
            st.info("👆 Enter your Groq API key to start generating flows")
            st.markdown("Get a free API key at: https://console.groq.com")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info(
        "ML-IDS uses machine learning to detect network intrusions. "
        "Uses 18 universal flow features - works with any network flow dataset. "
        "Automatically extracts features and applies StandardScaler. "
        "AI-powered analysis provides detailed insights into model performance and dataset characteristics."
    )


if __name__ == "__main__":
    main()
