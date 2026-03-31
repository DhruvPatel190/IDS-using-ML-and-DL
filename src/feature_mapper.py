"""
Feature Mapper for CICIDS2017 to Universal Features
Maps CICIDS2017 column names to universal feature names
"""

import pandas as pd
import numpy as np
from src.universal_features import UNIVERSAL_FEATURES

# CICIDS2017 to Universal Feature Mapping
CICIDS2017_FEATURE_MAP = {
    'Flow Duration': 'flow_duration',
    'Destination Port': 'dst_port',
    'Protocol': 'protocol',
    'Total Fwd Packets': 'tot_fwd_pkts',
    'Total Backward Packets': 'tot_bwd_pkts',
    'Flow Packets/s': 'flow_pkts_s',
    'Flow Bytes/s': 'flow_byts_s',
    'Fwd Packet Length Mean': 'fwd_pkt_len_mean',
    'Fwd Packet Length Max': 'fwd_pkt_len_max',
    'Bwd Packet Length Mean': 'bwd_pkt_len_mean',
    'Packet Length Mean': 'pkt_len_mean',
    'Packet Length Std': 'pkt_len_std',
    'Flow IAT Mean': 'flow_iat_mean',
    'Flow IAT Std': 'flow_iat_std',
    'Flow IAT Max': 'flow_iat_max',
    'SYN Flag Count': 'syn_flag_cnt',
    'ACK Flag Count': 'ack_flag_cnt',
    'RST Flag Count': 'rst_flag_cnt',
    'Average Packet Size': 'pkt_size_avg'
}


def map_cicids2017_features(df):
    """
    Map CICIDS2017 features to universal features
    
    Args:
        df: DataFrame with CICIDS2017 columns
        
    Returns:
        DataFrame with only universal features
    """
    # Select only the columns we need
    cicids_cols = list(CICIDS2017_FEATURE_MAP.keys())
    
    # Check which columns exist
    available_cols = [col for col in cicids_cols if col in df.columns]
    missing_cols = [col for col in cicids_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    # Select available columns
    df_selected = df[available_cols].copy()
    
    # Rename to universal names
    rename_map = {k: v for k, v in CICIDS2017_FEATURE_MAP.items() if k in available_cols}
    df_selected.rename(columns=rename_map, inplace=True)
    
    # Ensure all universal features are present
    for feature in UNIVERSAL_FEATURES:
        if feature not in df_selected.columns:
            print(f"Warning: Feature {feature} not found, filling with zeros")
            df_selected[feature] = 0
    
    # Return in correct order
    return df_selected[UNIVERSAL_FEATURES]


def get_universal_features_from_cicids2017(df):
    """
    Extract universal features from CICIDS2017 dataset
    
    Args:
        df: Full CICIDS2017 DataFrame
        
    Returns:
        X: Features (universal features only)
        y: Labels (binary)
    """
    # Map features
    X = map_cicids2017_features(df)
    
    # Get labels
    if 'binary_label' in df.columns:
        y = df['binary_label']
    elif 'Label' in df.columns:
        # Convert to binary: BENIGN=0, anything else=1
        y = (df['Label'] != 'BENIGN').astype(int)
    else:
        raise ValueError("No label column found")
    
    return X, y
