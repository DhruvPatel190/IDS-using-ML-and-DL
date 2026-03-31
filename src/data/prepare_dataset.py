"""
Dataset preparation script for CICIDS2017
Loads raw CSV files, preprocesses, and splits into train/val/test sets
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_RATIO, VAL_RATIO, 
    TEST_RATIO, RANDOM_STATE, LABEL_COLUMN
)


def load_cicids2017_data():
    """Load all CICIDS2017 CSV files from raw data directory"""
    print("Loading CICIDS2017 dataset...")
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")
    
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(RAW_DATA_DIR, csv_file)
        print(f"Reading {csv_file}...")
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"Total samples loaded: {len(data)}")
    return data


def preprocess_data(df):
    """Preprocess the dataset"""
    print("Preprocessing data...")
    
    # Standardize column names (lowercase, replace spaces with underscores)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Create binary label: 0 for BENIGN, 1 for any attack
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
    
    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with missing values
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows with missing values")
    
    # Remove duplicate rows
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {initial_rows - len(df)} duplicate rows")
    
    return df


def split_dataset(df):
    """Split dataset into train, validation, and test sets"""
    print("Splitting dataset...")
    
    # Separate features and labels
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: train and val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, 
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train, X_val, X_test):
    """Normalize features using StandardScaler"""
    print("Normalizing features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save processed datasets"""
    print("Saving processed datasets...")
    
    # Save as pickle files
    joblib.dump((X_train, y_train), os.path.join(PROCESSED_DATA_DIR, 'train.pkl'))
    joblib.dump((X_val, y_val), os.path.join(PROCESSED_DATA_DIR, 'val.pkl'))
    joblib.dump((X_test, y_test), os.path.join(PROCESSED_DATA_DIR, 'test.pkl'))
    
    print("Datasets saved successfully!")


def main():
    """Main execution function"""
    # Load data
    df = load_cicids2017_data()
    
    # Preprocess
    df = preprocess_data(df)
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    
    # Normalize features
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    
    # Save processed data
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\nDataset preparation completed!")
    print(f"Class distribution in training set:")
    print(f"  Benign: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
    print(f"  Malicious: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")


if __name__ == "__main__":
    main()
