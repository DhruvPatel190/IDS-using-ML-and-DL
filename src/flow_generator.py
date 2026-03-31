"""
Network Flow Generator using Groq LLM
Generates synthetic network flow data for IDS testing
Uses CICIDS2017 dataset as reference for realistic patterns
"""
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os


class FlowGenerator:
    """Generate synthetic network flows using CICIDS2017 as reference"""
    
    def __init__(self, api_key: str, reference_dataset: str = 'cicids2017_full.csv'):
        """Initialize with Groq API key and reference dataset"""
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        
        # Load reference dataset for sampling
        if os.path.exists(reference_dataset):
            print(f"Loading reference dataset: {reference_dataset}")
            self.ref_df = pd.read_csv(reference_dataset)
            self.ref_df['binary_label'] = self.ref_df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            
            # Separate benign and attack samples
            self.benign_samples = self.ref_df[self.ref_df['binary_label'] == 0]
            self.attack_samples = self.ref_df[self.ref_df['binary_label'] == 1]
            print(f"Loaded {len(self.benign_samples)} benign and {len(self.attack_samples)} attack samples")
        else:
            print(f"Warning: Reference dataset not found. Using rule-based generation.")
            self.ref_df = None
            self.benign_samples = None
            self.attack_samples = None
    
    def generate_scenario(self, scenario_type: str, count: int = 100) -> pd.DataFrame:
        """
        Generate flows for specific scenarios
        Samples from CICIDS2017 and adds small variations for realism
        Includes True_Label column: 0=Benign, 1=Attack
        """
        # Universal features we need
        feature_cols = [
            'Flow Duration', 'Destination Port', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Packets/s', 'Flow Bytes/s', 'Fwd Packet Length Mean', 'Fwd Packet Length Max',
            'Bwd Packet Length Mean', 'Packet Length Mean', 'Packet Length Std',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
            'SYN Flag Count', 'ACK Flag Count', 'RST Flag Count', 'Average Packet Size'
        ]
        
        flows = []
        
        if scenario_type == 'benign':
            # Sample benign flows
            flows = self._sample_flows(self.benign_samples, count, feature_cols, label=0)
        
        elif scenario_type == 'ddos':
            # Sample DDoS attacks
            ddos_samples = self.attack_samples[self.attack_samples['Label'].str.contains('DDoS', case=False, na=False)]
            if len(ddos_samples) > 0:
                flows = self._sample_flows(ddos_samples, count, feature_cols, label=1)
            else:
                # Fallback to any attack
                flows = self._sample_flows(self.attack_samples, count, feature_cols, label=1)
        
        elif scenario_type == 'port_scan':
            # Sample port scan attacks
            portscan_samples = self.attack_samples[self.attack_samples['Label'].str.contains('PortScan', case=False, na=False)]
            if len(portscan_samples) > 0:
                flows = self._sample_flows(portscan_samples, count, feature_cols, label=1)
            else:
                flows = self._sample_flows(self.attack_samples, count, feature_cols, label=1)
        
        elif scenario_type == 'brute_force':
            # Sample brute force attacks (FTP-Patator, SSH-Patator)
            brute_samples = self.attack_samples[
                self.attack_samples['Label'].str.contains('Patator', case=False, na=False)
            ]
            if len(brute_samples) > 0:
                flows = self._sample_flows(brute_samples, count, feature_cols, label=1)
            else:
                flows = self._sample_flows(self.attack_samples, count, feature_cols, label=1)
        
        else:  # mixed
            # 70% benign, 30% attacks
            benign_count = int(count * 0.7)
            attack_count = count - benign_count
            
            benign_flows = self._sample_flows(self.benign_samples, benign_count, feature_cols, label=0)
            attack_flows = self._sample_flows(self.attack_samples, attack_count, feature_cols, label=1)
            flows = benign_flows + attack_flows
        
        return pd.DataFrame(flows)
    
    def _sample_flows(self, source_df, count, feature_cols, label):
        """
        Sample flows from source dataset and add small random variations
        """
        if source_df is None or len(source_df) == 0:
            # Fallback to rule-based if no reference data
            return [self._generate_fallback_flow(label) for _ in range(count)]
        
        # Sample with replacement if needed
        sampled = source_df.sample(n=min(count, len(source_df)), replace=(count > len(source_df)))
        
        flows = []
        for _, row in sampled.iterrows():
            flow = {}
            for feat in feature_cols:
                if feat in row:
                    val = row[feat]
                    # Add small random variation (±10%)
                    if pd.notna(val) and val != np.inf and val != -np.inf:
                        variation = np.random.uniform(0.9, 1.1)
                        flow[feat] = val * variation
                    else:
                        flow[feat] = 0
                else:
                    flow[feat] = 0
            
            flow['True_Label'] = label
            flows.append(flow)
        
        return flows
    
    def _generate_fallback_flow(self, label):
        """Fallback rule-based generation if no reference data"""
        if label == 0:
            return self._generate_benign_flow()
        else:
            return self._generate_ddos_flow()
    
    def _generate_benign_flow(self):
        """Generate benign traffic pattern (fallback)"""
        port = np.random.choice([80, 443, 53, 8080, 25, 123], p=[0.4, 0.3, 0.15, 0.1, 0.03, 0.02])
        
        flow = {
            'Flow Duration': np.random.randint(100000, 10000000),
            'Destination Port': port,
            'Total Fwd Packets': np.random.randint(2, 50),
            'Total Backward Packets': np.random.randint(2, 50),
            'Flow Packets/s': np.random.uniform(10, 1000),
            'Flow Bytes/s': np.random.uniform(1000, 100000),
            'Fwd Packet Length Mean': np.random.uniform(200, 600),
            'Fwd Packet Length Max': np.random.randint(400, 1500),
            'Bwd Packet Length Mean': np.random.uniform(200, 600),
            'Packet Length Mean': np.random.uniform(200, 600),
            'Packet Length Std': np.random.uniform(100, 400),
            'Flow IAT Mean': np.random.uniform(50000, 5000000),
            'Flow IAT Std': np.random.uniform(10000, 3000000),
            'Flow IAT Max': np.random.randint(100000, 10000000),
            'SYN Flag Count': np.random.randint(0, 2),
            'ACK Flag Count': np.random.randint(10, 100),
            'RST Flag Count': np.random.randint(0, 2),
            'Average Packet Size': np.random.uniform(200, 600)
        }
        flow['True_Label'] = 0
        return flow
    
    def _generate_ddos_flow(self):
        """Generate DDoS attack pattern (fallback)"""
        flow = {
            'Flow Duration': np.random.randint(10000, 2000000),
            'Destination Port': np.random.choice([80, 443, 53]),
            'Total Fwd Packets': np.random.randint(100, 1000),
            'Total Backward Packets': np.random.randint(0, 20),
            'Flow Packets/s': np.random.uniform(5000, 50000),
            'Flow Bytes/s': np.random.uniform(100000, 2000000),
            'Fwd Packet Length Mean': np.random.uniform(50, 200),
            'Fwd Packet Length Max': np.random.randint(100, 500),
            'Bwd Packet Length Mean': np.random.uniform(20, 100),
            'Packet Length Mean': np.random.uniform(50, 200),
            'Packet Length Std': np.random.uniform(20, 150),
            'Flow IAT Mean': np.random.uniform(100, 50000),
            'Flow IAT Std': np.random.uniform(100, 30000),
            'Flow IAT Max': np.random.randint(1000, 200000),
            'SYN Flag Count': np.random.randint(1, 10),
            'ACK Flag Count': np.random.randint(20, 200),
            'RST Flag Count': np.random.randint(0, 5),
            'Average Packet Size': np.random.uniform(50, 200)
        }
        flow['True_Label'] = 1
        return flow
