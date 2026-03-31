"""
Model training script
Trains multiple ML and DL models on the preprocessed dataset
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE


def load_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    X_train, y_train = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'train.pkl'))
    X_val, y_val = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'val.pkl'))
    X_test, y_test = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'test.pkl'))
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and print metrics"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    print("\n" + "="*50)
    print("Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, "Random Forest")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    print("\n" + "="*50)
    print("Training XGBoost...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=1
    )
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, "XGBoost")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'xgboost.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, metrics


def train_catboost(X_train, y_train, X_test, y_test):
    """Train CatBoost classifier"""
    print("\n" + "="*50)
    print("Training CatBoost...")
    
    model = CatBoostClassifier(
        iterations=100,
        depth=10,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=50
    )
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, "CatBoost")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'catboost.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, metrics


def build_neural_network(input_dim):
    """Build a simple feedforward neural network"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Neural Network classifier"""
    print("\n" + "="*50)
    print("Training Neural Network...")
    
    model = build_neural_network(X_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=256,
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nNeural Network Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'neural_network.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, metrics


def main():
    """Main training function"""
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    results = {}
    
    # Train ML models
    _, results['RandomForest'] = train_random_forest(X_train, y_train, X_test, y_test)
    _, results['XGBoost'] = train_xgboost(X_train, y_train, X_test, y_test)
    _, results['CatBoost'] = train_catboost(X_train, y_train, X_test, y_test)
    
    # Train DL model
    _, results['NeuralNetwork'] = train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    
    # Save results
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'))
    print(f"\nResults saved to {os.path.join(MODELS_DIR, 'model_comparison.csv')}")


if __name__ == "__main__":
    main()
