import os
import numpy as np
import pandas as pd
from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

def generate_test_csv_files():
    """Generates synthetic flow CSV datasets for batch ingestion testing."""
    print("Generating synthetic CSV test datasets...")
    np.random.seed(2026)
    
    classes = [
        "Benign Normal Traffic",
        "DoS / DDoS Flood",
        "PortScan (Reconnaissance)",
        "Botnet Command & Control",
        "Infiltration & Exploit",
        "Web Attack (SQLi / XSS)",
        "Brute Force (SSH/FTP)",
        "Malware Data Exfiltration"
    ]
    
    num_features = 78
    feature_names = [
        "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
        "Fwd Packet Length Mean", "Fwd Packet Length Std", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
        "Flow IAT Min", "SYN Flag Count", "RST Flag Count", "FIN Flag Count",
        "Fwd Header Length"
    ] + [f"Feature_{i}" for i in range(15, num_features)]
    
    records = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        # Generate 60 records per class
        for _ in range(60):
            vec = np.random.normal(loc=0.1, scale=0.05, size=num_features)
            vec = np.clip(vec, 0, None)
            
            if class_idx == 0:  # Benign
                vec[0] = np.random.uniform(200, 1500)
                vec[1] = np.random.randint(15, 80)
                vec[2] = np.random.uniform(5000, 50000)
                vec[3] = np.random.uniform(300, 900)
                vec[11] = np.random.randint(1, 3)
            elif class_idx == 1:  # DoS / DDoS
                vec[0] = np.random.uniform(10, 80)
                vec[1] = np.random.randint(12000, 60000)
                vec[11] = np.random.randint(10000, 50000)
                vec[7] = np.random.uniform(0.0001, 0.0008)
            elif class_idx == 2:  # PortScan
                vec[0] = np.random.uniform(1.0, 4.0)
                vec[1] = np.random.randint(600, 2000)
                vec[11] = np.random.randint(500, 1800)
            elif class_idx == 3:  # Botnet
                vec[0] = np.random.uniform(1000, 6000)
                vec[2] = np.random.randint(64, 256)
                vec[3] = np.random.uniform(32, 64)
            elif class_idx == 4:  # Infiltration & Exploit
                vec[0] = np.random.uniform(1500, 9000)
                vec[3] = np.random.uniform(1100, 1480)
                vec[14] = np.random.randint(800, 2500)
            elif class_idx == 5:  # Web Attack
                vec[0] = np.random.uniform(300, 1500)
                vec[3] = np.random.uniform(950, 1400)
                vec[14] = np.random.randint(700, 1500)
            elif class_idx == 6:  # Brute Force
                vec[0] = np.random.uniform(400, 2500)
                vec[1] = np.random.randint(300, 900)
                vec[12] = np.random.randint(150, 600)
            elif class_idx == 7:  # Exfiltration
                vec[0] = np.random.uniform(3000, 12000)
                vec[2] = np.random.uniform(200000, 1500000)
                vec[3] = np.random.uniform(1250, 1460)
                
            records.append(vec)
            labels.append(class_name)
            
    df = pd.DataFrame(records, columns=feature_names)
    
    # Save test_batch_flows.csv (Without label for blind testing)
    df.to_csv("test_batch_flows.csv", index=False)
    print("Saved test_batch_flows.csv (480 flow samples)")
    
    # Save test_attack_dataset.csv (With ground truth label)
    df_labeled = df.copy()
    df_labeled["Ground_Truth_Label"] = labels
    df_labeled.to_csv("test_attack_dataset.csv", index=False)
    print("Saved test_attack_dataset.csv (labeled flow samples)")
    
    # Save a smaller 20-row sample_telemetry.csv
    sample_df = df_labeled.sample(n=20, random_state=42)
    sample_df.to_csv("sample_telemetry.csv", index=False)
    print("Saved sample_telemetry.csv (20 sample rows)")

def generate_test_pcap_file():
    """Generates synthetic PCAP file with Scapy packet flows."""
    print("Generating synthetic PCAP capture file (test_capture.pcap)...")
    packets = []
    
    # 1. Normal TLS/HTTP Flow
    src_ip = "192.168.1.105"
    dst_ip = "10.0.0.1"
    for i in range(15):
        pkt = Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=49152+i, dport=443, flags="S")
        packets.append(pkt)
        pkt_ack = Ether()/IP(src=dst_ip, dst=src_ip)/TCP(sport=443, dport=49152+i, flags="SA")
        packets.append(pkt_ack)
        pkt_data = Ether()/IP(src=src_ip, dst=dst_ip)/TCP(sport=49152+i, dport=443, flags="A")/Raw(b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n")
        packets.append(pkt_data)

    # 2. SYN Flood Attack Packets
    attacker_ip = "10.10.99.12"
    target_ip = "10.0.0.1"
    for i in range(150):
        pkt = Ether()/IP(src=attacker_ip, dst=target_ip)/TCP(sport=1024 + (i % 60000), dport=80, flags="S")
        packets.append(pkt)
        
    # 3. PortScan Sequence
    scanner_ip = "172.16.0.45"
    for port in range(20, 100):
        pkt = Ether()/IP(src=scanner_ip, dst=target_ip)/TCP(sport=55000, dport=port, flags="S")
        packets.append(pkt)
        
    wrpcap("test_capture.pcap", packets)
    print(f"Saved test_capture.pcap ({len(packets)} raw packets)")

if __name__ == "__main__":
    generate_test_csv_files()
    generate_test_pcap_file()
    print("All test data generation completed successfully!")
