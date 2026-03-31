"""
Universal Feature Set for Network Flow Analysis
Defines core features that exist across different network flow datasets
"""

# Universal feature set - core flow behavior features
UNIVERSAL_FEATURES = [
    'flow_duration',
    'dst_port',
    'protocol',
    'tot_fwd_pkts',
    'tot_bwd_pkts',
    'flow_pkts_s',
    'flow_byts_s',
    'fwd_pkt_len_mean',
    'fwd_pkt_len_max',
    'bwd_pkt_len_mean',
    'pkt_len_mean',
    'pkt_len_std',
    'flow_iat_mean',
    'flow_iat_std',
    'flow_iat_max',
    'syn_flag_cnt',
    'ack_flag_cnt',
    'rst_flag_cnt',
    'pkt_size_avg'
]

# Number of features
NUM_FEATURES = len(UNIVERSAL_FEATURES)

# Identifier columns to remove (case-insensitive patterns)
IDENTIFIER_COLUMNS = [
    'flow_id',
    'src_ip',
    'source_ip',
    'dst_ip',
    'destination_ip',
    'timestamp',
    'flow_start',
    'flow_end',
    'src_port',
    'source_port'
]

# Label column names (case-insensitive)
LABEL_COLUMNS = ['label', 'class', 'attack', 'category']

# Binary classification mapping
BENIGN_LABELS = ['benign', 'normal', '0', 0]
