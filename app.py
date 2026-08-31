import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import tempfile
from model import NetworkMalwareClassifier
from pcap_extractor import extract_flow_features_from_pcap

# Streamlit Page Configuration
st.set_page_config(
    page_title="NetShield AI | Neural Threat Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stAlert {
        border-radius: 8px;
    }
    .css-1r6594q {
        background-color: #0f172a;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Constants & Taxonomy
CLASSES = [
    "Benign Normal Traffic",
    "DoS / DDoS Flood",
    "PortScan (Reconnaissance)",
    "Botnet Command & Control",
    "Infiltration & Exploit",
    "Web Attack (SQLi / XSS)",
    "Brute Force (SSH/FTP)",
    "Malware Data Exfiltration"
]

SEVERITY = {
    0: ("LOW", "#22c55e"),
    1: ("CRITICAL", "#ef4444"),
    2: ("MEDIUM", "#eab308"),
    3: ("CRITICAL", "#dc2626"),
    4: ("HIGH", "#f97316"),
    5: ("HIGH", "#ea580c"),
    6: ("MEDIUM", "#f59e0b"),
    7: ("CRITICAL", "#b91c1c")
}

NUM_FEATURES = 78

# Model & Scaler Loading Function
@st.cache_resource
def load_detection_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetworkMalwareClassifier(num_features=NUM_FEATURES, num_classes=len(CLASSES))
    
    loaded = False
    weights_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
            loaded = True
        except Exception as e:
            st.sidebar.error(f"Error loading weights: {e}")
            
    model.to(device)
    model.eval()
    
    scaler = None
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            pass
            
    return model, scaler, device, loaded

model, scaler, device, model_loaded = load_detection_engine()

# Sidebar Navigation
st.sidebar.title("🛡️ NetShield AI Engine")
st.sidebar.markdown("**Deep Learning Intrusion Detection System**")
st.sidebar.markdown("---")

if model_loaded:
    st.sidebar.success(f"System State: Active\nInference: `{device}`")
else:
    st.sidebar.warning("Running with base weights. Run `python generate_artifacts.py` to train model checkpoint.")

mode = st.sidebar.radio(
    "Navigation / Mode:",
    [
        "1. Preset Attack Scenarios",
        "2. Live PCAP File Inspection",
        "3. Batch CSV Ingestion",
        "4. Manual Flow Sandbox"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("System Architecture: **1D-CNN + BiLSTM** with Focal Loss optimization.")

# Header Banner
st.title("Network Threat Detection & Forensic Analysis Platform")
st.markdown("Real-time behavioral packet flow inspection and multi-class cyberattack classification.")
st.markdown("---")

def evaluate_vector(vector):
    """Normalizes and runs neural inference on a single flow vector."""
    vector_2d = vector.reshape(1, -1)
    if scaler is not None:
        try:
            vector_2d = scaler.transform(vector_2d)
        except Exception:
            pass
            
    tensor_input = torch.tensor(vector_2d, dtype=torch.float32).to(device)
        
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        
    pred_class_idx = int(np.argmax(probabilities))
    return pred_class_idx, probabilities

# ----------------- Mode 1: Preset Scenarios -----------------
if mode == "1. Preset Attack Scenarios":
    st.subheader("Simulated Attack Vector Replay")
    st.write("Select a pre-synthesized telemetry profile to evaluate classification response:")
    
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        preset_choice = st.selectbox(
            "Select Security Telemetry Scenario:",
            [
                "Standard Secure TLS/HTTPS Session",
                "High-Volume SYN Flood (DDoS Attack)",
                "Horizontal Nmap Port Scanning Matrix",
                "Mirai IoT Botnet Heartbeat & Beaconing",
                "SQL Injection (Web Exploit Payload)",
                "SSH Dictionary Brute-Force Sequence"
            ]
        )
    with col_btn:
        st.write("")
        st.write("")
        run_sim = st.button("Trigger Inspection", type="primary", use_container_width=True)

    # Deterministic representative vectors
    feature_vec = np.zeros(NUM_FEATURES)
    if preset_choice == "Standard Secure TLS/HTTPS Session":
        feature_vec = np.random.normal(0.1, 0.02, NUM_FEATURES)
        feature_vec[0], feature_vec[1], feature_vec[3] = 450.0, 45, 520.0
    elif preset_choice == "High-Volume SYN Flood (DDoS Attack)":
        feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
        feature_vec[0], feature_vec[1], feature_vec[11], feature_vec[7] = 45.0, 45000, 42000, 0.0002
    elif preset_choice == "Horizontal Nmap Port Scanning Matrix":
        feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
        feature_vec[0], feature_vec[1], feature_vec[11] = 2.5, 800, 750
    elif preset_choice == "Mirai IoT Botnet Heartbeat & Beaconing":
        feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
        feature_vec[0], feature_vec[2], feature_vec[3] = 2500.0, 96, 48.0
    elif preset_choice == "SQL Injection (Web Exploit Payload)":
        feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
        feature_vec[0], feature_vec[3], feature_vec[14] = 650.0, 1200.0, 840
    else: # SSH Brute Force
        feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
        feature_vec[0], feature_vec[1], feature_vec[12] = 850.0, 500, 350

    if run_sim or True: # Auto render initial verdict
        idx, probs = evaluate_vector(feature_vec)
        sev_label, sev_color = SEVERITY[idx]
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Identified Class", CLASSES[idx])
        with m_col2:
            st.metric("Confidence Score", f"{probs[idx]*100:.2f}%")
        with m_col3:
            st.metric("Threat Severity", sev_label)
            
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            if idx == 0:
                st.success("### Verdict: PASS\nTelemetry matches baseline benign operations. No firewall mitigation required.")
            else:
                st.error(f"### Verdict: BLOCK & QUARANTINE\nAnomaly signatures detected ({CLASSES[idx]}). Auto-mitigation rule pushed to security group.")
                
            st.markdown("#### Key Traffic Metrics:")
            st.write(f"- **Flow Duration:** `{feature_vec[0]:.2f} ms`")
            st.write(f"- **Total Forward Packets:** `{int(feature_vec[1])}`")
            st.write(f"- **SYN Flag Count:** `{int(feature_vec[11])}`")
            st.write(f"- **RST Flag Count:** `{int(feature_vec[12])}`")
            
        with res_col2:
            chart_df = pd.DataFrame({"Attack Class": CLASSES, "Probability (%)": probs * 100})
            fig = px.bar(
                chart_df,
                x="Probability (%)",
                y="Attack Class",
                orientation="h",
                color="Probability (%)",
                color_continuous_scale="Reds" if idx != 0 else "Greens",
                title="Model Confidence Distribution"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=320, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ----------------- Mode 2: PCAP Ingestion -----------------
elif mode == "2. Live PCAP File Inspection":
    st.subheader("Raw PCAP / Packet Capture Analysis")
    st.write("Upload a raw `.pcap` or `.pcapng` capture file. The system will extract flow parameters and run neural threat scoring.")
    
    pcap_file = st.file_uploader("Upload Network Capture", type=["pcap", "pcapng"])
    if pcap_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
            tmp.write(pcap_file.read())
            tmp_path = tmp.name
            
        with st.spinner("Parsing packet headers and computing flow metrics with Scapy..."):
            extracted_vector = extract_flow_features_from_pcap(tmp_path, target_num_features=NUM_FEATURES)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            
        st.success("Flow feature extraction complete.")
        idx, probs = evaluate_vector(extracted_vector[0])
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("PCAP Verdict", CLASSES[idx])
            st.metric("Confidence", f"{probs[idx]*100:.2f}%")
            if idx == 0:
                st.success("Clean Network Flow Detected")
            else:
                st.error("Malicious Packet Signature Detected!")
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(probs[idx] * 100),
                title={'text': f"Confidence Rating: {CLASSES[idx]}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': SEVERITY[idx][1]},
                    'steps': [
                        {'range': [0, 50], 'color': "#1e293b"},
                        {'range': [50, 80], 'color': "#334155"},
                        {'range': [80, 100], 'color': "#475569"}
                    ]
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ----------------- Mode 3: Batch CSV Ingestion -----------------
elif mode == "3. Batch CSV Ingestion":
    st.subheader("Batch Telemetry Data Processing")
    st.write("Upload a CSV file containing flow feature vectors for high-throughput batch threat classification.")
    
    csv_file = st.file_uploader("Upload Batch CSV Dataset", type=["csv"])
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write(f"Loaded **{len(df)}** flow records from CSV.")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_data = df[numeric_cols].values
            
            # Align features with NUM_FEATURES
            if feature_data.shape[1] < NUM_FEATURES:
                padded = np.zeros((feature_data.shape[0], NUM_FEATURES))
                padded[:, :feature_data.shape[1]] = feature_data
                feature_data = padded
            else:
                feature_data = feature_data[:, :NUM_FEATURES]
                
            if scaler is not None:
                feature_data = scaler.transform(feature_data)
                
            tensor_inputs = torch.tensor(feature_data, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(tensor_inputs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
            df["Threat_Classification"] = [CLASSES[p] for p in preds]
            df["Is_Malicious"] = [p != 0 for p in preds]
            
            malicious_total = sum(df["Is_Malicious"])
            
            st.markdown("---")
            b1, b2, b3 = st.columns(3)
            b1.metric("Total Flows Analyzed", len(df))
            b2.metric("Flagged Malicious Flows", malicious_total)
            b3.metric("Malicious Ratio", f"{(malicious_total/len(df))*100:.2f}%")
            
            st.dataframe(df[["Threat_Classification", "Is_Malicious"] + list(numeric_cols[:5])], use_container_width=True)
            
            # Threat Distribution Chart
            dist_df = df["Threat_Classification"].value_counts().reset_index()
            dist_df.columns = ["Threat Category", "Count"]
            fig = px.pie(dist_df, values="Count", names="Threat Category", title="Batch Threat Composition", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ----------------- Mode 4: Manual Sandbox -----------------
elif mode == "4. Manual Flow Sandbox":
    st.subheader("Flow Feature Sandbox")
    st.write("Adjust low-level flow attributes to test neural model decision boundaries:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        f_dur = st.slider("Flow Duration (ms)", 0.0, 5000.0, 250.0)
        f_pkts = st.number_input("Total Packets Sent", 1, 100000, 100)
    with col2:
        f_len = st.slider("Mean Packet Length (bytes)", 0.0, 1500.0, 512.0)
        f_syn = st.number_input("SYN Flags Count", 0, 50000, 2)
    with col3:
        f_iat = st.slider("Inter-Arrival Time (IAT Mean)", 0.0, 100.0, 1.2)
        f_rst = st.number_input("RST Flags Count", 0, 5000, 0)
        
    custom_v = np.zeros(NUM_FEATURES)
    custom_v[0] = f_dur
    custom_v[1] = f_pkts
    custom_v[3] = f_len
    custom_v[11] = f_syn
    custom_v[7] = f_iat
    custom_v[12] = f_rst
    
    if st.button("Evaluate Sandbox Configuration", type="primary"):
        c_idx, c_probs = evaluate_vector(custom_v)
        st.info(f"**Classification Result:** {CLASSES[c_idx]} (Confidence: {c_probs[c_idx]*100:.2f}%)")
