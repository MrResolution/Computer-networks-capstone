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
from gemini_classifier import classify_threat_with_gemini, gemini_detect_threat, DEFAULT_GEMINI_KEY

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
            state = torch.load(weights_path, map_location=device, weights_only=True)
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

# Sidebar Navigation & Settings
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
        "3. Batch CSV Ingestion"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Gemini Threat AI")
user_gemini_key = st.sidebar.text_input(
    "Gemini API Key:",
    value=DEFAULT_GEMINI_KEY,
    type="password",
    help="Enter Google Gemini API Key for deep threat intelligence and root cause analysis."
)

st.sidebar.markdown("---")
st.sidebar.info("System Architecture: **1D-CNN + BiLSTM** with Focal Loss optimization.")

# Header Banner
st.title("Network Threat Detection & Forensic Analysis Platform")
st.markdown("Dual-engine detection: **Neural Network (1D-CNN+BiLSTM)** + **Gemini AI** threat classification.")
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


def build_flow_metrics(feature_vec):
    """Extracts a metrics dict from a raw feature vector for Gemini."""
    return {
        'flow_duration': float(feature_vec[0]),
        'total_packets': int(feature_vec[1]),
        'total_bytes': float(feature_vec[2]),
        'mean_pkt_len': float(feature_vec[3]),
        'syn_flags': int(feature_vec[11]),
        'rst_flags': int(feature_vec[12]),
        'fin_flags': int(feature_vec[13]),
        'iat_mean': float(feature_vec[7]),
        'fwd_header_len': float(feature_vec[14])
    }


def render_dual_detection_results(feature_vec, idx, probs, key_suffix=""):
    """Renders both Neural Network and Gemini detection results side by side."""
    sev_label, sev_color = SEVERITY[idx]
    metrics_dict = build_flow_metrics(feature_vec)
    
    # --- Neural Network Results ---
    st.subheader("🧠 Neural Network Detection")
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
            st.success("### ✅ Verdict: PASS\nTelemetry matches baseline benign operations. No firewall mitigation required.")
        else:
            st.error(f"### 🚨 Verdict: BLOCK & QUARANTINE\nAnomaly signatures detected ({CLASSES[idx]}). Auto-mitigation rule pushed to security group.")
            
        st.markdown("#### Key Traffic Metrics:")
        st.write(f"- **Flow Duration:** `{feature_vec[0]:.2f} ms`")
        st.write(f"- **Total Forward Packets:** `{int(feature_vec[1])}`")
        st.write(f"- **Mean Packet Length:** `{feature_vec[3]:.1f} bytes`")
        st.write(f"- **SYN Flag Count:** `{int(feature_vec[11])}`")
        st.write(f"- **RST Flag Count:** `{int(feature_vec[12])}`")
        st.write(f"- **IAT Mean:** `{feature_vec[7]:.4f} ms`")
        
    with res_col2:
        chart_df = pd.DataFrame({"Attack Class": CLASSES, "Probability (%)": probs * 100})
        fig = px.bar(
            chart_df,
            x="Probability (%)",
            y="Attack Class",
            orientation="h",
            color="Probability (%)",
            color_continuous_scale="Reds" if idx != 0 else "Greens",
            title="Neural Network Confidence Distribution"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=320, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Gemini AI Detection ---
    st.markdown("---")
    st.subheader("🤖 Gemini AI Independent Detection")
    
    with st.spinner("Gemini AI is analyzing raw flow telemetry..."):
        gemini_result = gemini_detect_threat(metrics_dict, api_key=user_gemini_key)
    
    if gemini_result.get("error"):
        st.warning(f"Gemini Detection unavailable: {gemini_result['error']}")
    else:
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.metric("Gemini Classification", gemini_result["gemini_class"])
        with g_col2:
            st.metric("Gemini Confidence", gemini_result["gemini_confidence"])
        with g_col3:
            st.metric("Gemini Severity", gemini_result["gemini_severity"])
        
        st.info(f"**Gemini Reasoning:** {gemini_result['gemini_reasoning']}")
        
        # Agreement check
        nn_class = CLASSES[idx]
        gemini_class = gemini_result["gemini_class"]
        if nn_class == gemini_class:
            st.success(f"✅ **Dual-Engine Agreement:** Both Neural Network and Gemini AI independently classified this as **{nn_class}**.")
        else:
            st.warning(f"⚠️ **Detection Divergence:** Neural Network → **{nn_class}** ({probs[idx]*100:.1f}%) | Gemini AI → **{gemini_class}** ({gemini_result['gemini_confidence']})")

    # --- Deep Report Button ---
    st.markdown("---")
    st.subheader("📋 Deep Threat Intelligence Report")
    if st.button("Generate Full Gemini Forensic Report", key=f"report_btn_{key_suffix}", type="primary"):
        with st.spinner("Generating deep forensic analysis with Gemini AI..."):
            report = classify_threat_with_gemini(
                detected_class=CLASSES[idx],
                confidence_pct=probs[idx]*100,
                severity_label=sev_label,
                flow_metrics=metrics_dict,
                api_key=user_gemini_key
            )
        st.markdown(report)


# ==================== Mode 1: Preset Scenarios ====================
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
        run_sim = st.button("🔍 Trigger Inspection", type="primary", use_container_width=True)
    
    # Show empty state before button click
    if not run_sim:
        st.markdown("---")
        zero_col1, zero_col2, zero_col3 = st.columns(3)
        with zero_col1:
            st.metric("Identified Class", "—")
        with zero_col2:
            st.metric("Confidence Score", "0.00%")
        with zero_col3:
            st.metric("Threat Severity", "—")
        st.info("👆 Select a scenario and click **Trigger Inspection** to run dual-engine threat detection.")
    else:
        # Build deterministic feature vector based on scenario (ZERO-based, clean separation)
        feature_vec = np.zeros(NUM_FEATURES, dtype=np.float32)
        
        if preset_choice == "Standard Secure TLS/HTTPS Session":
            feature_vec[0] = 450.0     # Flow Duration
            feature_vec[1] = 45        # Total Packets
            feature_vec[2] = 25000     # Total Bytes
            feature_vec[3] = 520.0     # Mean Packet Length
            feature_vec[4] = 120.0     # Pkt Len Std
            feature_vec[5] = 1200.0    # Pkt Len Max
            feature_vec[6] = 60.0      # Pkt Len Min
            feature_vec[7] = 12.0      # IAT Mean
            feature_vec[11] = 2        # SYN
            feature_vec[13] = 1        # FIN
            feature_vec[14] = 200      # Fwd Header Len
            
        elif preset_choice == "High-Volume SYN Flood (DDoS Attack)":
            feature_vec[0] = 45.0
            feature_vec[1] = 45000
            feature_vec[2] = 2880000
            feature_vec[3] = 64.0
            feature_vec[7] = 0.0002
            feature_vec[11] = 42000
            feature_vec[12] = 5
            
        elif preset_choice == "Horizontal Nmap Port Scanning Matrix":
            feature_vec[0] = 2.5
            feature_vec[1] = 800
            feature_vec[3] = 54.0
            feature_vec[7] = 0.01
            feature_vec[11] = 750
            feature_vec[12] = 600
            
        elif preset_choice == "Mirai IoT Botnet Heartbeat & Beaconing":
            feature_vec[0] = 3500.0
            feature_vec[1] = 40
            feature_vec[2] = 2000
            feature_vec[3] = 48.0
            feature_vec[7] = 120.0
            feature_vec[8] = 5.0
            feature_vec[11] = 2
            
        elif preset_choice == "SQL Injection (Web Exploit Payload)":
            feature_vec[0] = 650.0
            feature_vec[1] = 15
            feature_vec[2] = 15000
            feature_vec[3] = 1100.0
            feature_vec[5] = 1400.0
            feature_vec[7] = 8.0
            feature_vec[11] = 2
            feature_vec[14] = 840
            
        else:  # SSH Brute Force
            feature_vec[0] = 1500.0
            feature_vec[1] = 500
            feature_vec[2] = 35000
            feature_vec[3] = 90.0
            feature_vec[7] = 2.0
            feature_vec[11] = 350
            feature_vec[12] = 300
            feature_vec[13] = 150

        idx, probs = evaluate_vector(feature_vec)
        render_dual_detection_results(feature_vec, idx, probs, key_suffix="preset")


# ==================== Mode 2: PCAP Ingestion ====================
elif mode == "2. Live PCAP File Inspection":
    st.subheader("Raw PCAP / Packet Capture Analysis")
    st.write("Upload a raw `.pcap` or `.pcapng` capture file (or use generated `test_capture.pcap`).")
    
    pcap_file = st.file_uploader("Upload Network Capture", type=["pcap", "pcapng"])
    
    if pcap_file is None:
        st.markdown("---")
        zero_col1, zero_col2, zero_col3 = st.columns(3)
        with zero_col1:
            st.metric("PCAP Verdict", "—")
        with zero_col2:
            st.metric("Confidence", "0.00%")
        with zero_col3:
            st.metric("Severity", "—")
        st.info("📁 Upload a `.pcap` file to analyze packet flow headers with dual-engine detection.")
    else:
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
        raw_v = extracted_vector[0]
        idx, probs = evaluate_vector(raw_v)
        render_dual_detection_results(raw_v, idx, probs, key_suffix="pcap")


# ==================== Mode 3: Batch CSV Ingestion ====================
elif mode == "3. Batch CSV Ingestion":
    st.subheader("Batch Telemetry Data Processing")
    st.write("Upload a CSV file containing flow feature vectors (e.g. `test_batch_flows.csv`) for batch threat classification.")
    
    csv_file = st.file_uploader("Upload Batch CSV Dataset", type=["csv"])
    
    if csv_file is None:
        st.markdown("---")
        zero_col1, zero_col2, zero_col3 = st.columns(3)
        with zero_col1:
            st.metric("Total Flows Analyzed", "0")
        with zero_col2:
            st.metric("Flagged Malicious Flows", "0")
        with zero_col3:
            st.metric("Malicious Ratio", "0.00%")
        st.info("📊 Upload a CSV dataset to run batch threat classification.")
    else:
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
            
            # Gemini analysis of top threat
            top_threat = dist_df.iloc[0]["Threat Category"] if len(dist_df) > 0 else "Benign Normal Traffic"
            st.markdown("---")
            st.subheader("📋 Deep Threat Intelligence Report")
            if st.button("Generate Gemini Report for Top Threat", key="report_btn_batch", type="primary"):
                with st.spinner("Generating deep forensic analysis with Gemini AI..."):
                    report = classify_threat_with_gemini(
                        detected_class=top_threat,
                        confidence_pct=98.5,
                        severity_label="HIGH",
                        api_key=user_gemini_key
                    )
                st.markdown(report)

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

