"""
Generate sample network flow data for testing
This creates synthetic data when you don't have the actual CICIDS2017 dataset
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import RAW_DATA_DIR, FEATURE_COLUMNS


def generate_benign_flow():
    """Generate a benign network flow"""
    return {
        'dst_port': np.random.choice([80, 443, 22, 21, 25]),
        'protocol': np.random.choice([6, 17]),  # TCP or UDP
        'flow_duration': np.random.randint(1000, 10000000),
        'tot_fwd_pkts': np.random.randint(1, 100),
        'tot_bwd_pkts': np.random.randint(1, 100),
        'totlen_fwd_pkts': np.random.randint(100, 50000),
        'totlen_bwd_pkts': np.random.randint(100, 50000),
        'fwd_pkt_len_max': np.random.randint(100, 1500),
        'fwd_pkt_len_min': np.random.randint(20, 100),
        'fwd_pkt_len_mean': np.random.uniform(100, 800),
        'fwd_pkt_len_std': np.random.uniform(10, 200),
        'bwd_pkt_len_max': np.random.randint(100, 1500),
        'bwd_pkt_len_min': np.random.randint(20, 100),
        'bwd_pkt_len_mean': np.random.uniform(100, 800),
        'bwd_pkt_len_std': np.random.uniform(10, 200),
        'flow_byts_s': np.random.uniform(1000, 100000),
        'flow_pkts_s': np.random.uniform(10, 1000),
        'flow_iat_mean': np.random.uniform(100, 10000),
        'flow_iat_std': np.random.uniform(10, 5000),
        'flow_iat_max': np.random.randint(1000, 100000),
        'flow_iat_min': np.random.randint(1, 1000),
        'fwd_iat_tot': np.random.randint(1000, 1000000),
        'fwd_iat_mean': np.random.uniform(100, 10000),
        'fwd_iat_std': np.random.uniform(10, 5000),
        'fwd_iat_max': np.random.randint(1000, 100000),
        'fwd_iat_min': np.random.randint(1, 1000),
        'bwd_iat_tot': np.random.randint(1000, 1000000),
        'bwd_iat_mean': np.random.uniform(100, 10000),
        'bwd_iat_std': np.random.uniform(10, 5000),
        'bwd_iat_max': np.random.randint(1000, 100000),
        'bwd_iat_min': np.random.randint(1, 1000),
        'fwd_psh_flags': 0,
        'bwd_psh_flags': 0,
        'fwd_urg_flags': 0,
        'bwd_urg_flags': 0,
        'fwd_header_len': np.random.randint(20, 200),
        'bwd_header_len': np.random.randint(20, 200),
        'fwd_pkts_s': np.random.uniform(10, 500),
        'bwd_pkts_s': np.random.uniform(10, 500),
        'pkt_len_min': np.random.randint(20, 100),
        'pkt_len_max': np.random.randint(100, 1500),
        'pkt_len_mean': np.random.uniform(100, 800),
        'pkt_len_std': np.random.uniform(10, 200),
        'pkt_len_var': np.random.uniform(100, 40000),
        'fin_flag_cnt': np.random.randint(0, 2),
        'syn_flag_cnt': np.random.randint(0, 2),
        'rst_flag_cnt': 0,
        'psh_flag_cnt': np.random.randint(0, 5),
        'ack_flag_cnt': np.random.randint(1, 10),
        'urg_flag_cnt': 0,
        'cwe_flag_count': 0,
        'ece_flag_cnt': 0,
        'down_up_ratio': np.random.uniform(0, 5),
        'pkt_size_avg': np.random.uniform(100, 800),
        'fwd_seg_size_avg': np.random.uniform(100, 800),
        'bwd_seg_size_avg': np.random.uniform(100, 800),
        'fwd_byts_b_avg': 0,
        'fwd_pkts_b_avg': 0,
        'fwd_blk_rate_avg': 0,
        'bwd_byts_b_avg': 0,
        'bwd_pkts_b_avg': 0,
        'bwd_blk_rate_avg': 0,
        'subflow_fwd_pkts': np.random.randint(1, 100),
        'subflow_fwd_byts': np.random.randint(100, 50000),
        'subflow_bwd_pkts': np.random.randint(1, 100),
        'subflow_bwd_byts': np.random.randint(100, 50000),
        'init_fwd_win_byts': np.random.randint(1000, 65535),
        'init_bwd_win_byts': np.random.randint(1000, 65535),
        'fwd_act_data_pkts': np.random.randint(1, 50),
        'fwd_seg_size_min': np.random.randint(8, 100),
        'active_mean': np.random.uniform(100, 10000),
        'active_std': np.random.uniform(10, 5000),
        'active_max': np.random.randint(1000, 100000),
        'active_min': np.random.randint(1, 1000),
        'idle_mean': np.random.uniform(100, 10000),
        'idle_std': np.random.uniform(10, 5000),
        'idle_max': np.random.randint(1000, 100000),
        'idle_min': np.random.randint(1, 1000),
        'label': 'BENIGN'
    }


def generate_malicious_flow():
    """Generate a malicious network flow (simulated attack patterns)"""
    flow = generate_benign_flow()
    
    # Modify to simulate attack patterns
    attack_type = np.random.choice(['DoS', 'DDoS', 'PortScan', 'BruteForce'])
    
    if attack_type in ['DoS', 'DDoS']:
        # High packet rate, short duration
        flow['flow_pkts_s'] = np.random.uniform(10000, 100000)
        flow['tot_fwd_pkts'] = np.random.randint(1000, 10000)
        flow['pkt_len_mean'] = np.random.uniform(50, 200)
        flow['flow_duration'] = np.random.randint(100, 10000)
    
    elif attack_type == 'PortScan':
        # Many connections to different ports
        flow['dst_port'] = np.random.randint(1, 65535)
        flow['tot_fwd_pkts'] = np.random.randint(1, 10)
        flow['tot_bwd_pkts'] = 0
        flow['syn_flag_cnt'] = 1
        flow['ack_flag_cnt'] = 0
    
    elif attack_type == 'BruteForce':
        # Repeated connection attempts
        flow['dst_port'] = np.random.choice([22, 21, 3389])
        flow['tot_fwd_pkts'] = np.random.randint(5, 50)
        flow['flow_duration'] = np.random.randint(1000, 100000)
    
    flow['label'] = attack_type
    return flow


def generate_sample_dataset(n_samples=10000, malicious_ratio=0.3):
    """Generate a sample dataset"""
    print(f"Generating {n_samples} sample flows...")
    
    n_malicious = int(n_samples * malicious_ratio)
    n_benign = n_samples - n_malicious
    
    flows = []
    
    # Generate benign flows
    for _ in range(n_benign):
        flows.append(generate_benign_flow())
    
    # Generate malicious flows
    for _ in range(n_malicious):
        flows.append(generate_malicious_flow())
    
    df = pd.DataFrame(flows)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """Generate and save sample dataset"""
    print("Generating sample CICIDS2017-like dataset...")
    print("Note: This is synthetic data for testing purposes only!")
    
    # Generate dataset
    df = generate_sample_dataset(n_samples=10000, malicious_ratio=0.3)
    
    # Save to raw data directory
    output_path = os.path.join(RAW_DATA_DIR, 'sample_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nSample dataset saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print("\nYou can now run: python src/data/prepare_dataset.py")


if __name__ == "__main__":
    main()
