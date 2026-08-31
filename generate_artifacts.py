import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from train import train_network_model

def generate_synthetic_ids_data(num_samples_per_class=300, num_features=78):
    """
    Synthesizes realistic statistical flow telemetry across 8 target security classes.
    """
    np.random.seed(42)
    X = []
    y = []
    
    for c in range(8):
        for _ in range(num_samples_per_class):
            vec = np.random.normal(loc=0.1, scale=0.05, size=num_features)
            
            if c == 0:  # Benign Normal Traffic
                vec[0] = np.random.uniform(100, 1000)  # Flow Duration
                vec[1] = np.random.randint(10, 100)    # Total Packets
                vec[3] = np.random.uniform(200, 800)   # Mean Packet Length
            elif c == 1: # DoS / DDoS Flood
                vec[0] = np.random.uniform(10, 100)
                vec[1] = np.random.randint(10000, 50000) # Extreme packet count
                vec[11] = np.random.randint(8000, 45000) # Massive SYN flags
                vec[7] = np.random.uniform(0.0001, 0.001) # Low IAT
            elif c == 2: # PortScan Reconnaissance
                vec[0] = np.random.uniform(1.0, 5.0)
                vec[1] = np.random.randint(500, 1500)
                vec[11] = np.random.randint(450, 1400) # High SYN scan count
            elif c == 3: # Botnet C&C
                vec[0] = np.random.uniform(500, 5000)
                vec[2] = np.random.randint(64, 128)   # Small heartbeat size
                vec[3] = np.random.uniform(32, 64)
            elif c == 4: # Infiltration & Exploit
                vec[0] = np.random.uniform(1000, 8000)
                vec[3] = np.random.uniform(1000, 1500) # Large exploit payload
                vec[14] = np.random.randint(500, 2000)
            elif c == 5: # Web Attack (SQLi / XSS)
                vec[0] = np.random.uniform(200, 1200)
                vec[3] = np.random.uniform(900, 1400)
                vec[14] = np.random.randint(600, 1200)
            elif c == 6: # Brute Force (SSH/FTP)
                vec[0] = np.random.uniform(300, 2000)
                vec[12] = np.random.randint(100, 500) # High RST count
                vec[1] = np.random.randint(200, 800)
            elif c == 7: # Malware Data Exfiltration
                vec[0] = np.random.uniform(2000, 10000)
                vec[2] = np.random.uniform(100000, 1000000) # Large outbound bytes
                vec[3] = np.random.uniform(1200, 1450)
                
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
    X, y = generate_synthetic_ids_data(num_samples_per_class=400, num_features=78)
    
    # Train / Val Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
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
        epochs=15,
        batch_size=128,
        save_path="model_weights.pth"
    )
    print("Pre-training complete! Artifacts (model_weights.pth, scaler.pkl) ready.")

if __name__ == "__main__":
    main()
