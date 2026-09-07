# NetShield AI — Development Log & Milestone History

This document logs architectural changes, design iterations, model enhancements, and bug fixes for the **NetShield AI** Network Intrusion Detection System.

---

## 📅 Milestone Log

### Phase 1: Core Neural Network & Training Pipeline
- **Initial Baseline Architecture**:
  - Implemented 1D Convolutional Neural Network (CNN) for spatial feature extraction on tabular CIC-IDS flow features.
  - Implemented standard 78-feature preprocessing pipeline with `StandardScaler` normalization.
  - Built training script (`train.py`) supporting multi-class cross-entropy loss and Adam optimizer.

- **Hybrid 1D-CNN + BiLSTM Model (`model.py`)**:
  - Upgraded model architecture to combine 1D spatial convolutions with a 2-layer Bidirectional LSTM.
  - Added a direct Fully-Connected (FC) path for raw tabular feature retention.
  - Fusion layer combines all 3 representations (320 dimensions total) before classification head.
  - Improved multi-class accuracy across minority attack classes (Botnet, Infiltration, Heartbleed).

---

### Phase 2: Live PCAP Parsing & Feature Extraction
- **Scapy Integration (`pcap_extractor.py`)**:
  - Created packet inspection pipeline to parse live or offline `.pcap` files.
  - Extracted flow metrics: flow duration, packet counts, length statistics (mean, std, min, max), inter-arrival times (IAT), TCP flags (SYN, RST, FIN), and header lengths.
  - Standardized feature vector mapping to match the 78-feature CIC-IDS format.

- **Synthetic Artifact Generators (`generate_artifacts.py`, `generate_test_data.py`)**:
  - Implemented dataset generator for synthetic flow vectors representing 8 attack classes.
  - Exported pre-trained weights (`model_weights.pth`) and standard scaler (`scaler.pkl`).
  - Added sample `.pcap` and `.csv` generator for rapid system verification without external dataset downloads.

---

### Phase 3: Dual User Interfaces
- **Streamlit Web Dashboard (`app.py`)**:
  - Built web interface for network traffic analysis.
  - Added interactive preset vector evaluator, live PCAP dropzone, and batch CSV analysis.
  - Integrated Plotly charts for confidence distributions and attack breakdown visuals.

- **PyQt5 Desktop Application (`desktop_app.py`)**:
  - Built native GUI desktop application for offline network threat evaluation.
  - Added sidebar navigation, custom dark-mode theme, metric cards, and integrated Matplotlib canvases.
  - Created standalone launcher script (`run_desktop.sh`).

- **Gemini Threat Intelligence (`gemini_classifier.py`)**:
  - Added Gemini LLM fallback/analysis module to generate threat reports and remediation steps for detected network anomalies.

---

### Phase 4: Maintenance & Repository Hygiene
- **`.gitignore` Fix**:
  - Cleaned up `.gitignore` to prevent agent and temporary AI state files (`.agents/`, `.gemini/`, `skills-lock.json`, `metadata.json`) from cluttering the repository.
- **Repository Verification**:
  - Added `verify_system.py` to validate environment dependencies, PyTorch weights, and classifier modules prior to execution.
  - Updated documentation (`README.md`, `devlog.md`).

---

## 🛠️ Summary of Key Technical Changes

| Component | Initial Implementation | Current Architecture | Reason for Upgrade |
| :--- | :--- | :--- | :--- |
| **Model** | Simple 1D-CNN | Hybrid 1D-CNN + BiLSTM + FC Direct | Better temporal sequence & tabular feature capture |
| **Input Format** | Static CSV vector | CSV + Live PCAP via Scapy | Real-world packet capture compatibility |
| **UI** | CLI / Streamlit only | Streamlit + Native PyQt5 Desktop GUI | Accessible web & offline desktop operation |
| **Threat Explanation** | Hardcoded labels | Gemini LLM Threat Intelligence Integration | Rich contextual incident response reports |
