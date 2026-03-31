"""Configuration file for ML-IDS project"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_STATE = 42

# Model parameters
ML_MODELS = ['RandomForest', 'XGBoost', 'CatBoost']
DL_MODELS = ['NeuralNetwork', 'LSTM']

# Universal Feature Set (18 features - exact CICIDS2017 column names)
# These match the exact column names from CICIDS2017 dataset
FEATURE_COLUMNS = [
    'Flow Duration',
    'Destination Port',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow Packets/s',
    'Flow Bytes/s',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Max',
    'Bwd Packet Length Mean',
    'Packet Length Mean',
    'Packet Length Std',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'SYN Flag Count',
    'ACK Flag Count',
    'RST Flag Count',
    'Average Packet Size'
]

LABEL_COLUMN = 'label'
