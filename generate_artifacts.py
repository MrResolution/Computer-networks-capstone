import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from train import train_network_model

def generate_synthetic_ids_data(num_samples_per_class=500, num_features=78):
    """
    Synthesizes realistic statistical flow telemetry across 8 target security classes.
    Each class has strongly separated feature signatures to ensure clean classification.
    """
    np.random.seed(42)
    X = []
    y = []
    
    for c in range(8):
        for _ in range(num_samples_per_class):
            # Start with a ZERO vector — only set features that matter for each class
            vec = np.zeros(num_features, dtype=np.float32)
            
            if c == 0:  # Benign Normal Traffic
                vec[0] = np.random.uniform(200, 2000)     # Flow Duration (moderate)
                vec[1] = np.random.randint(10, 150)        # Total Packets (low-moderate)
                vec[2] = np.random.uniform(5000, 80000)    # Total bytes (moderate)
                vec[3] = np.random.uniform(200, 800)       # Mean Packet Length (normal range)
                vec[4] = np.random.uniform(50, 200)        # Pkt Len Std
                vec[5] = np.random.uniform(500, 1400)      # Pkt Len Max
                vec[6] = np.random.uniform(40, 100)        # Pkt Len Min
                vec[7] = np.random.uniform(5.0, 50.0)      # IAT Mean (normal)
                vec[8] = np.random.uniform(1.0, 20.0)      # IAT Std
                vec[11] = np.random.randint(1, 5)           # SYN flags (very low)
                vec[12] = 0                                  # RST flags (none)
                vec[13] = np.random.randint(1, 3)           # FIN flags (normal close)
                vec[14] = np.random.uniform(100, 400)       # Fwd Header Length
                
            elif c == 1:  # DoS / DDoS Flood
                vec[0] = np.random.uniform(5, 100)          # Flow Duration (very short burst)
                vec[1] = np.random.randint(10000, 60000)    # Total Packets (MASSIVE)
                vec[2] = np.random.uniform(500000, 3000000) # Total bytes (huge)
                vec[3] = np.random.uniform(40, 80)          # Mean Pkt Len (small SYN pkts)
                vec[7] = np.random.uniform(0.0001, 0.005)   # IAT Mean (near-zero, saturating)
                vec[11] = np.random.randint(8000, 55000)    # SYN flags (EXTREME)
                vec[12] = np.random.randint(0, 20)          # RST (very few)
                
            elif c == 2:  # PortScan (Reconnaissance)
                vec[0] = np.random.uniform(0.5, 8.0)        # Flow Duration (very short probes)
                vec[1] = np.random.randint(400, 2000)        # Moderate packet count
                vec[3] = np.random.uniform(40, 70)           # Small probing packets
                vec[7] = np.random.uniform(0.001, 0.05)      # Low IAT (rapid scanning)
                vec[11] = np.random.randint(400, 1800)       # High SYN (connection attempts)
                vec[12] = np.random.randint(200, 1500)       # High RST (ports closed → reset)
                
            elif c == 3:  # Botnet Command & Control
                vec[0] = np.random.uniform(1000, 8000)       # Long duration (persistent C2)
                vec[1] = np.random.randint(20, 80)           # Low packet count (heartbeats)
                vec[2] = np.random.uniform(500, 5000)        # Small total bytes
                vec[3] = np.random.uniform(30, 70)           # Small heartbeat packets
                vec[7] = np.random.uniform(50.0, 200.0)      # High IAT (periodic beaconing)
                vec[8] = np.random.uniform(1.0, 10.0)        # Low IAT Std (regular interval)
                vec[11] = np.random.randint(1, 5)            # Normal SYN
                
            elif c == 4:  # Infiltration & Exploit
                vec[0] = np.random.uniform(2000, 12000)      # Long duration (exploitation phase)
                vec[1] = np.random.randint(50, 300)           # Moderate packets
                vec[2] = np.random.uniform(50000, 500000)     # Large payload transfer
                vec[3] = np.random.uniform(1000, 1480)        # Large exploit payloads
                vec[5] = np.random.uniform(1400, 1500)        # Max pkt near MTU
                vec[7] = np.random.uniform(10.0, 80.0)        # Moderate IAT
                vec[14] = np.random.randint(800, 3000)        # Large fwd header (exploit headers)
                
            elif c == 5:  # Web Attack (SQLi / XSS)
                vec[0] = np.random.uniform(100, 1500)         # Medium duration
                vec[1] = np.random.randint(5, 50)             # Few packets (crafted requests)
                vec[2] = np.random.uniform(2000, 30000)       # Moderate bytes
                vec[3] = np.random.uniform(800, 1400)         # Large payloads (injection strings)
                vec[5] = np.random.uniform(1200, 1500)        # Max pkt large
                vec[7] = np.random.uniform(1.0, 20.0)         # Normal-ish IAT
                vec[11] = np.random.randint(1, 5)             # Normal SYN
                vec[14] = np.random.randint(500, 1500)        # Elevated header length
                
            elif c == 6:  # Brute Force (SSH/FTP)
                vec[0] = np.random.uniform(500, 5000)         # Medium-long duration
                vec[1] = np.random.randint(200, 1000)          # Many connection attempts
                vec[2] = np.random.uniform(10000, 60000)       # Moderate bytes
                vec[3] = np.random.uniform(60, 150)            # Small auth packets
                vec[7] = np.random.uniform(0.5, 5.0)           # Fast retries
                vec[11] = np.random.randint(100, 800)          # High SYN (many connections)
                vec[12] = np.random.randint(100, 700)          # High RST (auth failures)
                vec[13] = np.random.randint(50, 400)           # Many FIN (closed connections)
                
            elif c == 7:  # Malware Data Exfiltration
                vec[0] = np.random.uniform(5000, 30000)        # Very long duration (slow siphon)
                vec[1] = np.random.randint(100, 500)            # Moderate packets
                vec[2] = np.random.uniform(500000, 5000000)     # MASSIVE outbound bytes
                vec[3] = np.random.uniform(1200, 1480)          # Large packets (data transfer)
                vec[5] = np.random.uniform(1450, 1500)          # Max pkt at MTU
                vec[7] = np.random.uniform(20.0, 100.0)         # Moderate IAT (steady stream)
                vec[11] = np.random.randint(1, 5)               # Normal SYN
                vec[13] = np.random.randint(1, 3)               # Normal FIN
                
            # Add small noise to non-zero features only (preserves zero-base separation)
            noise = np.random.normal(0, 0.01, num_features).astype(np.float32)
            mask = vec != 0
            vec[mask] += vec[mask] * noise[mask]
            
            X.append(vec)
            y.append(c)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def main():
    print("Generating synthetic network flow training dataset...")
    X, y = generate_synthetic_ids_data(num_samples_per_class=600, num_features=78)
    
    # Train / Val Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
    
    print("Fitting RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    joblib.dump(scaler, "scaler.pkl")
    print("Saved scaler to scaler.pkl")
    
    print("Training NetShield AI Neural Network Model...")
    train_network_model(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        num_classes=8,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        save_path="model_weights.pth"
    )
    print("Pre-training complete! Artifacts (model_weights.pth, scaler.pkl) ready.")

if __name__ == "__main__":
    main()
