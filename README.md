# ML-IDS: Machine Learning Intrusion Detection System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Machine Learning-based Intrusion Detection System featuring 24 diverse models (ML + DL) with 99%+ accuracy on CICIDS2017 dataset. Includes AI-powered analysis using Groq LLM for explainable results.

![ML-IDS Demo](https://img.shields.io/badge/Demo-Streamlit-red)

## 🎯 Key Features

- **24 ML/DL Models**: Ensemble of supervised and anomaly detection models
- **99%+ Accuracy**: Achieves 99.98% accuracy on CICIDS2017 test data
- **Universal Features**: Uses only 18 core network flow features (dataset-agnostic)
- **Interactive UI**: Streamlit-based web interface with progressive results
- **AI-Powered Analysis**: Groq LLM provides detailed insights and explanations
- **Synthetic Data Generation**: AI generates test flows from CICIDS2017 patterns
- **Real-time Capable**: <1ms inference per flow

## 📊 Models Implemented

### ML Supervised (9 models)
- Logistic Regression
- Random Forest ⭐ (99.98% accuracy)
- Histogram Gradient Boosting
- Naive Bayes
- Decision Tree ⭐ (99.95% accuracy)
- K-Nearest Neighbors
- Linear SVM
- Extra Trees ⭐ (99.97% accuracy)
- Balanced Random Forest

### ML Anomaly Detection (6 models)
- Isolation Forest
- Elliptic Envelope
- PCA Reconstruction
- kNN Anomaly Detection
- HBOS
- COPOD

### DL Supervised (5 models)
- Multi-Layer Perceptron (MLP)
- Transformer
- Vanilla LSTM
- Stacked LSTM
- Bidirectional LSTM

### DL Anomaly Detection (3 models)
- AutoEncoder
- LSTM AutoEncoder
- Denoising AutoEncoder

### Specialized (1 model)
- TabNet (Attention-based tabular learning)

## 🚀 Quick Start

### Prerequisites

- Python 3.11 (required for TensorFlow 2.21)
- Windows OS (batch scripts provided)
- 8GB+ RAM recommended
- Groq API key (optional, for AI analysis)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ml-ids.git
cd ml-ids
```

2. **Run the app** (first time will auto-setup)
```bash
run.bat
```

That's it! The script will:
- Check for Python 3.11
- Create virtual environment (first time only)
- Install all dependencies (first time only)
- Launch the Streamlit app at `http://localhost:8501`

**First run**: Takes 5-10 minutes (installing packages)  
**Subsequent runs**: Starts immediately

### Optional: Download CICIDS2017 Dataset
- Download from: [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- Place `cicids2017_full.csv` in project root
- Or use the provided test sample (already included)

## 📖 Usage

### Option 1: Upload Dataset
1. Open the app → **Upload Dataset** tab
2. Upload any CSV file with network flow data
3. (Optional) Enter Groq API key for AI analysis
4. Click **Analyze Dataset**
5. View progressive results from all 24 models
6. Read AI-powered insights

### Option 2: AI Generate Flows
1. Open the app → **AI Generate Flows** tab
2. Enter your Groq API key
3. Select scenario: benign, ddos, port_scan, brute_force, or mixed
4. Generate flows (samples from CICIDS2017 with variations)
5. Click **Analyze Generated Flows**
6. View accuracy vs ground truth
7. Read detailed AI analysis

## 🏗️ Project Structure

```
ml-ids/
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── config.py                 # Configuration & 18 universal features
│   ├── universal_features.py     # Feature definitions
│   ├── feature_mapper.py         # Maps dataset columns to universal names
│   ├── flow_generator.py         # AI flow generation
│   └── report_generator.py       # LLM-powered analysis
├── models/                        # Trained models (not in repo, train yourself)
├── notebooks/
│   ├── ML_Models_Training.ipynb  # Train 15 ML models
│   └── DL_Models_Training.ipynb  # Train 9 DL models
├── data/
│   ├── raw/                       # Raw datasets (not in repo)
│   └── processed/                 # Processed data (not in repo)
├── requirements.txt               # All dependencies
├── run.bat                        # One-click setup & run
├── README.md                      # This file
└── QUICK_START.md                 # Quick start guide
```

## 🎓 Training Models

### Train ML Models
1. Ensure `cicids2017_full.csv` is in project root
2. Open Jupyter: `run_jupyter_venv.bat`
3. Open `notebooks/ML_Models_Training.ipynb`
4. Run all cells
5. Models saved to `models/` directory

### Train DL Models
1. Open `notebooks/DL_Models_Training.ipynb`
2. Run all cells
3. DL models saved to `models/` directory

**Training Time**:
- ML models: 5-15 minutes
- DL models: 30-90 minutes
- Total: ~2 hours on standard laptop

## 📊 Performance

### CICIDS2017 Test Data (10,000 flows)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 99.98% | 99.96% | 99.97% | 99.97% |
| Extra Trees | 99.97% | 99.95% | 99.96% | 99.96% |
| Decision Tree | 99.95% | 99.93% | 99.94% | 99.94% |
| Histogram GB | 99.94% | 99.92% | 99.93% | 99.93% |
| Logistic Regression | 88.00% | 92.00% | 45.00% | 60.00% |
| Naive Bayes | 34.00% | 40.00% | 30.00% | 34.00% |

### Why Some Models Fail
- **Logistic Regression/Linear SVM**: Assume linear separability (attacks are non-linear)
- **Naive Bayes**: Assumes feature independence (network features are correlated)
- **Tree Models Excel**: Capture non-linear patterns and feature interactions

## 🌐 Universal Feature Set (18 Features)

Our system uses only 18 core features that exist in ANY network flow dataset:

**Flow Characteristics**: Flow Duration, Destination Port, Total Fwd Packets, Total Backward Packets

**Rate Metrics**: Flow Packets/s, Flow Bytes/s

**Packet Size**: Fwd Packet Length Mean, Fwd Packet Length Max, Bwd Packet Length Mean, Packet Length Mean, Packet Length Std

**Inter-Arrival Time**: Flow IAT Mean, Flow IAT Std, Flow IAT Max

**TCP Flags**: SYN Flag Count, ACK Flag Count, RST Flag Count

**Aggregate**: Average Packet Size

## 🚧 Limitations & Challenges

### Cross-Dataset Performance
- **CICIDS2017 test**: 99%+ accuracy ✓
- **CICIDS2018 test**: 50-70% accuracy ✗
- **Reason**: Domain shift (different network, different attacks, different distributions)

### Live Deployment
- Requires network-specific fine-tuning
- Collect 1-2 weeks baseline from target network
- Retrain models on target network data
- Expected accuracy after fine-tuning: 95-98%

### Why Live Detection Needs Adaptation
1. **Network Environment Mismatch**: Every network has unique "normal" behavior
2. **Temporal Drift**: 2017 training data vs 2026 live traffic
3. **Attack Evolution**: Modern attacks differ from CICIDS2017 attacks
4. **Feature Distribution Shift**: Different networks have different feature ranges

## 🛠️ How to Deploy on Your Network

### Step 1: Collect Baseline (2 weeks)
- Capture all network traffic using TCPDump
- Convert to flow features using CICFlowMeter
- Label as "BENIGN"

### Step 2: Generate Attack Data
- Run controlled penetration tests
- Simulate DDoS, port scans, brute force
- Label as "ATTACK"

### Step 3: Retrain Models
```python
# Combine your data + CICIDS2017
df_combined = pd.concat([your_network_data, cicids2017_data])

# Train models
model.fit(df_combined)
```

### Step 4: Deploy
- Use TCPDump → CICFlowMeter → Models pipeline
- Expected accuracy: 95-98% on your network

## 🤖 AI Integration

### Flow Generation
- Samples from CICIDS2017 dataset
- Adds ±10% random variation
- Preserves statistical relationships
- Includes ground truth labels

### LLM Analysis (Groq Llama 3.3 70B)
Provides:
- Executive summary
- Dataset characteristics
- Model performance explanations
- Why models succeed/fail
- Feature importance insights
- Actionable recommendations

## 📚 Documentation

- **[PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md)**: Comprehensive project documentation
  - Complete technical details
  - Challenges and solutions
  - Model performance analysis
  - Live deployment guide
  - Q&A section

## 🔧 Technology Stack

- **Python 3.11**: Core language
- **TensorFlow 2.21**: Deep learning framework
- **PyTorch 2.11**: TabNet implementation
- **scikit-learn**: ML algorithms
- **Streamlit 1.55**: Web interface
- **LangChain + Groq**: AI analysis
- **Plotly**: Interactive visualizations
- **PyOD**: Anomaly detection
- **imbalanced-learn**: Handle class imbalance

## 📄 Requirements

See `requirements.txt` for complete list. Key dependencies:
- tensorflow==2.21.0
- torch==2.11.0
- streamlit==1.55.0
- scikit-learn
- pandas
- numpy
- plotly
- langchain
- langchain-groq


## 🙏 Acknowledgments

- **CICIDS2017 Dataset**: Canadian Institute for Cybersecurity, University of New Brunswick
- **Groq**: For providing fast LLM inference
- **Open Source Community**: For the amazing ML/DL libraries

