# NetShield AI: Deep Learning Network Malware & Threat Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-41CD52.svg)](https://www.qt.io/)

NetShield AI is a state-of-the-art Network Intrusion Detection System (NIDS) designed to identify, classify, and analyze network threats in real time. It uses a **Hybrid 1D-CNN + BiLSTM Deep Learning Model** that processes 78 statistical network flow metrics (CIC-IDS benchmark standard) across dual interfaces: an interactive **Streamlit Web Application** and a **PyQt5 Desktop GUI Application**.

---

## 🌟 Key Features

- **Hybrid Deep Neural Network**: Merges 1D Convolutional Neural Networks (spatial feature extraction), Bidirectional Long Short-Term Memory networks (sequential flow dynamics), and direct tabular feature feedforward paths.
- **PCAP Packet Analysis**: Parses raw network packet captures (`.pcap`) on the fly using Scapy to derive 78 statistical network flow metrics.
- **8-Class Threat Classification**:
  1. `BENIGN` — Standard authorized network traffic
  2. `DDoS-UDP` — High-rate UDP flood Denial-of-Service attacks
  3. `PortScan` — Host/Port enumeration and reconnaissance scans
  4. `Botnet C2` — Botnet command and control communication loops
  5. `BruteForce` — SSH/FTP password spraying and credential brute-forcing
  6. `WebAttack` — Web application vectors (SQLi, XSS, Command Injection)
  7. `Infiltration` — Internal lateral movement and privilege escalation
  8. `Heartbleed` — OpenSSL buffer over-read exploitation
- **Dual User Interfaces**:
  - **Streamlit Web Interface** (`app.py`): Web dashboard featuring interactive Plotly visualizations, live PCAP parsing, and CSV batch processing.
  - **PyQt5 Desktop Application** (`desktop_app.py`): Native desktop application with dark-mode UI, preset threat triggers, PCAP upload, and CSV flow batch scoring.
- **Gemini AI Threat Intelligence Integration**: Automated contextual threat explanations for identified malicious attack vectors.
- **Automated Synthetic Data & Artifact Generators**: Tools to bootstrap training datasets, pre-calculate feature scalers, and build sample PCAP files.

---

## 🏗️ System Architecture

```
                       ┌───────────────────────────────────────┐
                       │  Network Flow Input / PCAP File       │
                       └───────────────────┬───────────────────┘
                                           │
                                           ▼
                       ┌───────────────────────────────────────┐
                       │ Scapy Extractor / Feature Preprocessor│
                       │     (78 Standard CIC-IDS Features)    │
                       └───────────────────┬───────────────────┘
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             │                             │                             │
             ▼                             ▼                             ▼
   ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
   │  1D-CNN Branch    │         │  BiLSTM Branch    │         │ Direct FC Branch  │
   │ Spatial Patterns  │         │ Sequential Flow   │         │ Tabular Features  │
   └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
             │                             │                             │
             └─────────────────────────────┼─────────────────────────────┘
                                           │
                                           ▼
                       ┌───────────────────────────────────────┐
                       │ Multi-Branch Feature Fusion (320-dim) │
                       └───────────────────┬───────────────────┘
                                           │
                                           ▼
                       ┌───────────────────────────────────────┐
                       │ Threat Classification Head (8 Classes)│
                       └───────────────────┬───────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
         ┌─────────────────────────┐               ┌─────────────────────────┐
         │  Streamlit Web Dashboard│               │  PyQt5 Desktop GUI      │
         └─────────────────────────┘               └─────────────────────────┘
```

---

## 🛠️ Project Structure

```
Computer-networks-capstone/
├── app.py                 # Streamlit Web Dashboard application
├── desktop_app.py         # PyQt5 Desktop Application interface
├── model.py               # NetworkMalwareClassifier model architecture (1D-CNN + BiLSTM)
├── pcap_extractor.py      # Real-time PCAP parsing & feature extractor (Scapy)
├── gemini_classifier.py   # LLM threat intelligence & explanation module
├── generate_artifacts.py  # Dataset generator, model training, scaler exporter
├── generate_test_data.py # Sample PCAP & flow CSV generator
├── train.py               # PyTorch training pipeline
├── verify_system.py       # Diagnostic system verification script
├── run_desktop.sh         # Launch script for PyQt5 Desktop App
├── requirements.txt       # Python dependency specifications
└── .gitignore             # Git ignore configuration
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- `pcap` capture permissions (if running PCAP extraction live)

### 2. Clone and Setup Environment
```bash
git clone https://github.com/MrResolution/Computer-networks-capstone.git
cd Computer-networks-capstone

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Generate Model Weights & Test Artifacts
Generate model weights (`model_weights.pth`), scaler (`scaler.pkl`), test dataset (`test_attack_dataset.csv`), test flows (`test_batch_flows.csv`), and sample PCAPs (`test_capture.pcap`):

```bash
python generate_artifacts.py
python generate_test_data.py
```

### 4. Verify System Setup
Run the automated verification script to confirm all dependencies, weights, and models are healthy:
```bash
python verify_system.py
```

---

## 🚀 Running the Applications

### Option A: Streamlit Web Dashboard
Launch the interactive web interface:
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

### Option B: PyQt5 Desktop GUI
Launch the native desktop window application:
```bash
./run_desktop.sh
# or
python desktop_app.py
```

---

## 📊 Model & Features Summary

| Component | Description |
| :--- | :--- |
| **Input Dimensionality** | 78 continuous numerical flow features |
| **1D-CNN Path** | `Conv1d(1->64)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Conv1d(64->128)` $\rightarrow$ `AdaptiveAvgPool1d(1)` |
| **BiLSTM Path** | 2-layer Bidirectional LSTM (`hidden_size=64`, `dropout=0.2`) |
| **FC Direct Path** | 2-layer Linear dense network (`78 -> 128 -> 64`) |
| **Fusion Layer** | Concatenation of CNN (128) + BiLSTM (128) + FC (64) = 320 dimensions |
| **Classes** | 8 Multi-class labels (BENIGN + 7 Attack Families) |

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
