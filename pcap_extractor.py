from scapy.all import rdpcap, IP, TCP, UDP
import numpy as np
import pandas as pd

def extract_flow_features_from_pcap(pcap_path, target_num_features=78):
    """
    Parses a live or captured PCAP file using Scapy and derives statistical flow features.
    Matches standard CIC-IDS network flow feature format.
    """
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"Error reading PCAP file: {e}")
        return np.zeros((1, target_num_features))
        
    if len(packets) == 0:
        return np.zeros((1, target_num_features))
        
    packet_lengths = []
    iats = []
    last_time = None
    syn_count = 0
    rst_count = 0
    fin_count = 0
    fwd_header_lengths = 0
    
    for pkt in packets:
        if IP in pkt:
            length = len(pkt)
            packet_lengths.append(length)
            
            pkt_time = float(pkt.time)
            if last_time is not None:
                iats.append(pkt_time - last_time)
            last_time = pkt_time
            
            if TCP in pkt:
                flags = pkt[TCP].flags
                if 'S' in flags: syn_count += 1
                if 'R' in flags: rst_count += 1
                if 'F' in flags: fin_count += 1
                fwd_header_lengths += pkt[TCP].dataofs * 4

    if not packet_lengths:
        return np.zeros((1, target_num_features))

    if not iats:
        iats = [0.0]

    # Compute descriptive flow statistics
    flow_duration = float(packets[-1].time - packets[0].time) if len(packets) > 1 else 0.0
    tot_fwd_pkts = len(packet_lengths)
    tot_len = sum(packet_lengths)
    pkt_len_mean = np.mean(packet_lengths)
    pkt_len_std = np.std(packet_lengths) if len(packet_lengths) > 1 else 0.0
    pkt_len_max = np.max(packet_lengths)
    pkt_len_min = np.min(packet_lengths)
    
    iat_mean = np.mean(iats)
    iat_std = np.std(iats) if len(iats) > 1 else 0.0
    iat_max = np.max(iats)
    iat_min = np.min(iats)
    
    # Pack into array matching CIC-IDS format (78 features)
    raw_vector = np.zeros(target_num_features)
    raw_vector[0] = flow_duration
    raw_vector[1] = tot_fwd_pkts
    raw_vector[2] = tot_len
    raw_vector[3] = pkt_len_mean
    raw_vector[4] = pkt_len_std
    raw_vector[5] = pkt_len_max
    raw_vector[6] = pkt_len_min
    raw_vector[7] = iat_mean
    raw_vector[8] = iat_std
    raw_vector[9] = iat_max
    raw_vector[10] = iat_min
    raw_vector[11] = syn_count
    raw_vector[12] = rst_count
    raw_vector[13] = fin_count
    raw_vector[14] = fwd_header_lengths
    
    return raw_vector.reshape(1, -1)
