import torch
import numpy as np
import os
import joblib
from model import NetworkMalwareClassifier
from pcap_extractor import extract_flow_features_from_pcap

def test_system():
    print("=== NetShield AI Automated System Verification ===")
    
    # 1. Test Model Forward Pass
    model = NetworkMalwareClassifier(num_features=78, num_classes=8)
    dummy_input = torch.randn(4, 78)
    output = model(dummy_input)
    assert output.shape == (4, 8), f"Expected shape (4, 8), got {output.shape}"
    print("[✓] Model forward pass test passed. Output shape:", output.shape)
    
    # 2. Test Artifact Existence
    assert os.path.exists("model_weights.pth"), "model_weights.pth missing"
    assert os.path.exists("scaler.pkl"), "scaler.pkl missing"
    print("[✓] Pre-trained weights and RobustScaler artifact check passed.")
    
    # 3. Test Scaler Transformation
    scaler = joblib.load("scaler.pkl")
    scaled_vector = scaler.transform(np.zeros((1, 78)))
    assert scaled_vector.shape == (1, 78)
    print("[✓] RobustScaler transformation test passed.")
    
    # 4. Test PyQt5 Standalone GUI Window Creation (Offscreen)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5.QtWidgets import QApplication
    from desktop_app import NetShieldDesktopApp

    app = QApplication.instance() or QApplication([])
    window = NetShieldDesktopApp()
    assert window.windowTitle().startswith("NetShield AI")
    print("[✓] Standalone PyQt5 desktop GUI initialization test passed.")

    print("\nAll automated verification checks PASSED successfully!")

if __name__ == "__main__":
    test_system()
