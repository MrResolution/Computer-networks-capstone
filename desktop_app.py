import sys
import os
import tempfile
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider, QDoubleSpinBox, QSpinBox,
    QStackedWidget, QListWidget, QListWidgetItem, QFrame, QProgressBar,
    QGroupBox, QGridLayout, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model import NetworkMalwareClassifier
from pcap_extractor import extract_flow_features_from_pcap

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

# Global QSS Theme (Dark Glassmorphism)
DARK_QSS = """
QMainWindow {
    background-color: #0b0f19;
}
QWidget {
    color: #f8fafc;
    font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 13px;
}
QFrame#SidebarFrame {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}
QListWidget {
    background-color: transparent;
    border: none;
    outline: none;
}
QListWidget::item {
    padding: 12px 16px;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 4px;
}
QListWidget::item:hover {
    background-color: #1e293b;
    color: #f8fafc;
}
QListWidget::item:selected {
    background-color: #3b82f6;
    color: #ffffff;
}
QFrame#Card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
}
QLabel#HeaderTitle {
    font-size: 22px;
    font-weight: bold;
    color: #f8fafc;
}
QLabel#HeaderSubtitle {
    font-size: 13px;
    color: #94a3b8;
}
QLabel#MetricTitle {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: bold;
}
QLabel#MetricValue {
    font-size: 20px;
    font-weight: bold;
    color: #38bdf8;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 10px 18px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:pressed {
    background-color: #1e40af;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px;
    color: #f8fafc;
}
QComboBox::drop-down {
    border: none;
}
QTableWidget {
    background-color: #0f172a;
    border: 1px solid #334155;
    gridline-color: #1e293b;
    border-radius: 8px;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 8px;
    font-weight: bold;
    border: none;
}
QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
}
QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 5px;
}
"""

# Matplotlib Dark Canvas Component
class DarkMatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0f172a')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0f172a')
        super(DarkMatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def plot_bar_chart(self, classes, probabilities, main_idx):
        self.axes.clear()
        colors = ['#ef4444' if i != 0 else '#22c55e' for i in range(len(classes))]
        y_pos = np.arange(len(classes))
        
        bars = self.axes.barh(y_pos, probabilities * 100, color=colors, height=0.6)
        self.axes.set_yticks(y_pos)
        self.axes.set_yticklabels(classes, color='#94a3b8', fontsize=9)
        self.axes.invert_yaxis()
        self.axes.set_xlabel("Probability (%)", color='#94a3b8', fontsize=10)
        self.axes.set_title("Model Threat Confidence Distribution", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
        
        self.axes.tick_params(colors='#94a3b8')
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_color('#334155')
        self.axes.spines['left'].set_color('#334155')
        
        self.fig.tight_layout()
        self.draw()

    def plot_pie_chart(self, distribution_df):
        self.axes.clear()
        labels = distribution_df["Threat Category"]
        counts = distribution_df["Count"]
        
        colors = ['#22c55e', '#ef4444', '#eab308', '#dc2626', '#f97316', '#ea580c', '#f59e0b', '#b91c1c']
        colors = colors[:len(labels)]
        
        wedges, texts, autotexts = self.axes.pie(
            counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops=dict(color="#94a3b8", fontsize=9)
        )
        for autotext in autotexts:
            autotext.set_color('#ffffff')
            autotext.set_weight('bold')
            
        self.axes.set_title("Batch Threat Distribution Breakdown", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
        self.fig.tight_layout()
        self.draw()

# Worker Thread for PCAP Extraction
class PCAPWorker(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, pcap_path):
        super().__init__()
        self.pcap_path = pcap_path

    def run(self):
        try:
            vector = extract_flow_features_from_pcap(self.pcap_path, target_num_features=NUM_FEATURES)
            self.finished.emit(vector)
        except Exception as e:
            self.error.emit(str(e))

# Worker Thread for CSV Ingestion
class CSVWorker(QThread):
    finished = pyqtSignal(pd.DataFrame, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

    def run(self):
        try:
            df = pd.read_csv(self.csv_path)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_data = df[numeric_cols].values
            
            if feature_data.shape[1] < NUM_FEATURES:
                padded = np.zeros((feature_data.shape[0], NUM_FEATURES))
                padded[:, :feature_data.shape[1]] = feature_data
                feature_data = padded
            else:
                feature_data = feature_data[:, :NUM_FEATURES]
                
            self.finished.emit(df, feature_data)
        except Exception as e:
            self.error.emit(str(e))


class NetShieldDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NetShield AI | Neural Threat Classifier & PCAP Forensic System")
        self.resize(1200, 800)
        
        # Load Model & Scaler Engine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NetworkMalwareClassifier(num_features=NUM_FEATURES, num_classes=len(CLASSES))
        self.scaler = None
        self.model_loaded = False
        self._load_engine()

        # Build UI Structure
        self._init_ui()

    def _load_engine(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "model_weights.pth")
        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading model weights: {e}")

        self.model.to(self.device)
        self.model.eval()

        scaler_path = os.path.join(base_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Error loading scaler: {e}")

    def evaluate_vector(self, vector):
        vector_2d = vector.reshape(1, -1)
        if self.scaler is not None:
            try:
                vector_2d = self.scaler.transform(vector_2d)
            except Exception:
                pass

        tensor_input = torch.tensor(vector_2d, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_input)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class_idx = int(np.argmax(probabilities))
        return pred_class_idx, probabilities

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ----------------- Left Sidebar -----------------
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("SidebarFrame")
        sidebar_frame.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(20, 24, 20, 24)

        title_label = QLabel("🛡️ NetShield AI")
        title_label.setObjectName("HeaderTitle")
        subtitle_label = QLabel("Deep Learning Intrusion Detection")
        subtitle_label.setObjectName("HeaderSubtitle")

        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(subtitle_label)
        sidebar_layout.addSpacing(20)

        # Engine Status Card
        status_card = QFrame()
        status_card.setObjectName("Card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 12, 12, 12)

        st_title = QLabel("ENGINE STATE")
        st_title.setObjectName("MetricTitle")
        st_val = QLabel("ACTIVE" if self.model_loaded else "BASE WEIGHTS")
        st_val.setStyleSheet("color: #22c55e; font-weight: bold;" if self.model_loaded else "color: #eab308; font-weight: bold;")
        st_dev = QLabel(f"Inference Device: {self.device.type.upper()}")
        st_dev.setStyleSheet("color: #94a3b8; font-size: 11px;")

        status_layout.addWidget(st_title)
        status_layout.addWidget(st_val)
        status_layout.addWidget(st_dev)
        sidebar_layout.addWidget(status_card)
        sidebar_layout.addSpacing(20)

        # Navigation List
        self.nav_list = QListWidget()
        nav_items = [
            "1. Preset Attack Scenarios",
            "2. Live PCAP Inspection",
            "3. Batch CSV Ingestion",
            "4. Manual Flow Sandbox"
        ]
        for item in nav_items:
            self.nav_list.addItem(QListWidgetItem(item))
        self.nav_list.setCurrentRow(0)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

        sidebar_layout.addWidget(self.nav_list)
        sidebar_layout.addStretch()

        arch_label = QLabel("Architecture: 1D-CNN + BiLSTM\nLoss: Multi-Class Focal Loss")
        arch_label.setStyleSheet("color: #64748b; font-size: 11px;")
        sidebar_layout.addWidget(arch_label)

        main_layout.addWidget(sidebar_frame)

        # ----------------- Right Main View -----------------
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 24, 24, 24)

        header_banner = QLabel("Network Threat Detection & Forensic Analysis Platform")
        header_banner.setObjectName("HeaderTitle")
        sub_banner = QLabel("Real-time behavioral packet flow inspection and multi-class cyberattack classification.")
        sub_banner.setObjectName("HeaderSubtitle")

        content_layout.addWidget(header_banner)
        content_layout.addWidget(sub_banner)
        content_layout.addSpacing(16)

        # Stacked Pages
        self.stacked_widget = QStackedWidget()
        self._build_page1_preset()
        self._build_page2_pcap()
        self._build_page3_csv()
        self._build_page4_sandbox()

        content_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(content_widget, stretch=1)

        # Trigger initial page update
        self._evaluate_preset_scenario()

    def _on_nav_changed(self, index):
        self.stacked_widget.setCurrentIndex(index)

    # ----------------- Page 1: Preset Scenarios -----------------
    def _build_page1_preset(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Selector Row
        sel_card = QFrame()
        sel_card.setObjectName("Card")
        sel_layout = QHBoxLayout(sel_card)

        lbl = QLabel("Select Security Telemetry Scenario:")
        lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Standard Secure TLS/HTTPS Session",
            "High-Volume SYN Flood (DDoS Attack)",
            "Horizontal Nmap Port Scanning Matrix",
            "Mirai IoT Botnet Heartbeat & Beaconing",
            "SQL Injection (Web Exploit Payload)",
            "SSH Dictionary Brute-Force Sequence"
        ])
        
        run_btn = QPushButton("Trigger Inspection")
        run_btn.clicked.connect(self._evaluate_preset_scenario)

        sel_layout.addWidget(lbl)
        sel_layout.addWidget(self.preset_combo, stretch=1)
        sel_layout.addWidget(run_btn)
        layout.addWidget(sel_card)

        # Metric Cards Row
        metrics_layout = QHBoxLayout()
        self.p1_m_class = self._create_metric_card("IDENTIFIED CLASS", "---")
        self.p1_m_conf = self._create_metric_card("CONFIDENCE SCORE", "---")
        self.p1_m_sev = self._create_metric_card("THREAT SEVERITY", "---")

        metrics_layout.addWidget(self.p1_m_class)
        metrics_layout.addWidget(self.p1_m_conf)
        metrics_layout.addWidget(self.p1_m_sev)
        layout.addLayout(metrics_layout)

        # Verdict & Chart Section
        bottom_layout = QHBoxLayout()
        
        # Left Details Card
        self.p1_verdict_card = QFrame()
        self.p1_verdict_card.setObjectName("Card")
        v_layout = QVBoxLayout(self.p1_verdict_card)

        self.p1_verdict_lbl = QLabel("Verdict Summary")
        self.p1_verdict_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.p1_details_lbl = QLabel("Traffic Telemetry Details...")
        self.p1_details_lbl.setStyleSheet("color: #cbd5e1; font-size: 13px; line-height: 1.5;")
        
        v_layout.addWidget(self.p1_verdict_lbl)
        v_layout.addWidget(self.p1_details_lbl)
        v_layout.addStretch()

        # Right Chart
        self.p1_canvas = DarkMatplotlibCanvas(width=6, height=4)

        bottom_layout.addWidget(self.p1_verdict_card, stretch=1)
        bottom_layout.addWidget(self.p1_canvas, stretch=1)
        layout.addLayout(bottom_layout)

        self.stacked_widget.addWidget(page)

    def _evaluate_preset_scenario(self):
        choice = self.preset_combo.currentText()
        feature_vec = np.zeros(NUM_FEATURES)
        
        if choice == "Standard Secure TLS/HTTPS Session":
            feature_vec = np.random.normal(0.1, 0.02, NUM_FEATURES)
            feature_vec[0], feature_vec[1], feature_vec[3] = 450.0, 45, 520.0
        elif choice == "High-Volume SYN Flood (DDoS Attack)":
            feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
            feature_vec[0], feature_vec[1], feature_vec[11], feature_vec[7] = 45.0, 45000, 42000, 0.0002
        elif choice == "Horizontal Nmap Port Scanning Matrix":
            feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
            feature_vec[0], feature_vec[1], feature_vec[11] = 2.5, 800, 750
        elif choice == "Mirai IoT Botnet Heartbeat & Beaconing":
            feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
            feature_vec[0], feature_vec[2], feature_vec[3] = 2500.0, 96, 48.0
        elif choice == "SQL Injection (Web Exploit Payload)":
            feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
            feature_vec[0], feature_vec[3], feature_vec[14] = 650.0, 1200.0, 840
        else: # SSH Brute Force
            feature_vec = np.random.normal(0.1, 0.05, NUM_FEATURES)
            feature_vec[0], feature_vec[1], feature_vec[12] = 850.0, 500, 350

        idx, probs = self.evaluate_vector(feature_vec)
        sev_label, sev_color = SEVERITY[idx]

        self._update_metric_card(self.p1_m_class, "IDENTIFIED CLASS", CLASSES[idx])
        self._update_metric_card(self.p1_m_conf, "CONFIDENCE SCORE", f"{probs[idx]*100:.2f}%")
        self._update_metric_card(self.p1_m_sev, "THREAT SEVERITY", sev_label, color=sev_color)

        if idx == 0:
            self.p1_verdict_card.setStyleSheet("QFrame#Card { border: 2px solid #22c55e; background-color: #064e3b; }")
            self.p1_verdict_lbl.setText("Verdict: PASS (Normal Traffic)")
            self.p1_verdict_lbl.setStyleSheet("color: #4ade80; font-size: 16px; font-weight: bold;")
        else:
            self.p1_verdict_card.setStyleSheet("QFrame#Card { border: 2px solid #ef4444; background-color: #7f1d1d; }")
            self.p1_verdict_lbl.setText(f"Verdict: BLOCK & QUARANTINE ({CLASSES[idx]})")
            self.p1_verdict_lbl.setStyleSheet("color: #fca5a5; font-size: 16px; font-weight: bold;")

        details_text = f"""
<b>Key Flow Telemetry Metrics:</b><br/>
• <b>Flow Duration:</b> {feature_vec[0]:.2f} ms<br/>
• <b>Total Forward Packets:</b> {int(feature_vec[1])}<br/>
• <b>Mean Packet Length:</b> {feature_vec[3]:.1f} bytes<br/>
• <b>SYN Flag Count:</b> {int(feature_vec[11])}<br/>
• <b>RST Flag Count:</b> {int(feature_vec[12])}<br/>
• <b>Inter-Arrival Time (IAT):</b> {feature_vec[7]:.4f} ms
        """
        self.p1_details_lbl.setText(details_text)
        self.p1_canvas.plot_bar_chart(CLASSES, probs, idx)

    # ----------------- Page 2: PCAP Inspection -----------------
    def _build_page2_pcap(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        upload_card = QFrame()
        upload_card.setObjectName("Card")
        up_layout = QHBoxLayout(upload_card)

        self.p2_file_lbl = QLabel("No PCAP file loaded")
        self.p2_file_lbl.setStyleSheet("color: #94a3b8; font-style: italic;")

        select_btn = QPushButton("📁 Browse PCAP File")
        select_btn.clicked.connect(self._browse_pcap_file)

        up_layout.addWidget(select_btn)
        up_layout.addWidget(self.p2_file_lbl, stretch=1)
        layout.addWidget(upload_card)

        self.p2_progress = QProgressBar()
        self.p2_progress.setVisible(False)
        layout.addWidget(self.p2_progress)

        p2_metrics_layout = QHBoxLayout()
        self.p2_m_class = self._create_metric_card("PCAP VERDICT", "---")
        self.p2_m_conf = self._create_metric_card("CONFIDENCE RATING", "---")
        self.p2_m_sev = self._create_metric_card("SEVERITY LEVEL", "---")

        p2_metrics_layout.addWidget(self.p2_m_class)
        p2_metrics_layout.addWidget(self.p2_m_conf)
        p2_metrics_layout.addWidget(self.p2_m_sev)
        layout.addLayout(p2_metrics_layout)

        bottom_layout = QHBoxLayout()
        self.p2_details_card = QFrame()
        self.p2_details_card.setObjectName("Card")
        d_layout = QVBoxLayout(self.p2_details_card)
        self.p2_details_lbl = QLabel("Upload a .pcap file to analyze packet flow headers.")
        d_layout.addWidget(self.p2_details_lbl)
        d_layout.addStretch()

        self.p2_canvas = DarkMatplotlibCanvas(width=6, height=4)
        bottom_layout.addWidget(self.p2_details_card, stretch=1)
        bottom_layout.addWidget(self.p2_canvas, stretch=1)
        layout.addLayout(bottom_layout)

        self.stacked_widget.addWidget(page)

    def _browse_pcap_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Network Packet Capture", "", "PCAP Files (*.pcap *.pcapng)")
        if file_path:
            self.p2_file_lbl.setText(f"Loaded: {os.path.basename(file_path)}")
            self.p2_progress.setRange(0, 0)
            self.p2_progress.setVisible(True)

            self.pcap_worker = PCAPWorker(file_path)
            self.pcap_worker.finished.connect(self._on_pcap_extracted)
            self.pcap_worker.error.connect(self._on_pcap_error)
            self.pcap_worker.start()

    def _on_pcap_extracted(self, vector):
        self.p2_progress.setVisible(False)
        idx, probs = self.evaluate_vector(vector[0])
        sev_label, sev_color = SEVERITY[idx]

        self._update_metric_card(self.p2_m_class, "PCAP VERDICT", CLASSES[idx])
        self._update_metric_card(self.p2_m_conf, "CONFIDENCE RATING", f"{probs[idx]*100:.2f}%")
        self._update_metric_card(self.p2_m_sev, "SEVERITY LEVEL", sev_label, color=sev_color)

        self.p2_canvas.plot_bar_chart(CLASSES, probs, idx)
        
        status_str = "<b>Clean Flow Identified</b>" if idx == 0 else f"<b style='color:#ef4444;'>Malicious Signature Flagged: {CLASSES[idx]}</b>"
        self.p2_details_lbl.setText(f"""
<h3>Analysis Complete</h3>
{status_str}<br/><br/>
<b>Extracted Vector Summary:</b><br/>
• Flow Duration: {vector[0][0]:.2f} ms<br/>
• Total Packets: {int(vector[0][1])}<br/>
• Total Bytes: {int(vector[0][2])}<br/>
• SYN Count: {int(vector[0][11])}<br/>
• RST Count: {int(vector[0][12])}
        """)

    def _on_pcap_error(self, err_msg):
        self.p2_progress.setVisible(False)
        QMessageBox.critical(self, "PCAP Parsing Error", f"Failed to parse PCAP file:\n{err_msg}")

    # ----------------- Page 3: CSV Ingestion -----------------
    def _build_page3_csv(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        up_card = QFrame()
        up_card.setObjectName("Card")
        u_layout = QHBoxLayout(up_card)

        self.p3_file_lbl = QLabel("No CSV dataset loaded")
        self.p3_file_lbl.setStyleSheet("color: #94a3b8; font-style: italic;")

        select_btn = QPushButton("📊 Browse CSV Dataset")
        select_btn.clicked.connect(self._browse_csv_file)

        u_layout.addWidget(select_btn)
        u_layout.addWidget(self.p3_file_lbl, stretch=1)
        layout.addWidget(up_card)

        p3_metrics_layout = QHBoxLayout()
        self.p3_m_total = self._create_metric_card("TOTAL FLOWS ANALYZED", "0")
        self.p3_m_mal = self._create_metric_card("FLAGGED MALICIOUS", "0")
        self.p3_m_ratio = self._create_metric_card("MALICIOUS RATIO", "0.00%")

        p3_metrics_layout.addWidget(self.p3_m_total)
        p3_metrics_layout.addWidget(self.p3_m_mal)
        p3_metrics_layout.addWidget(self.p3_m_ratio)
        layout.addLayout(p3_metrics_layout)

        table_chart_layout = QHBoxLayout()
        self.p3_table = QTableWidget()
        self.p3_table.setColumnCount(4)
        self.p3_table.setHorizontalHeaderLabels(["Row", "Predicted Classification", "Is Malicious", "Confidence"])
        self.p3_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.p3_canvas = DarkMatplotlibCanvas(width=5, height=4)
        table_chart_layout.addWidget(self.p3_table, stretch=2)
        table_chart_layout.addWidget(self.p3_canvas, stretch=1)
        layout.addLayout(table_chart_layout)

        self.stacked_widget.addWidget(page)

    def _browse_csv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Telemetry CSV Dataset", "", "CSV Files (*.csv)")
        if file_path:
            self.p3_file_lbl.setText(f"Loaded: {os.path.basename(file_path)}")
            self.csv_worker = CSVWorker(file_path)
            self.csv_worker.finished.connect(self._on_csv_processed)
            self.csv_worker.error.connect(self._on_csv_error)
            self.csv_worker.start()

    def _on_csv_processed(self, df, feature_data):
        if self.scaler is not None:
            try:
                feature_data = self.scaler.transform(feature_data)
            except Exception:
                pass

        tensor_inputs = torch.tensor(feature_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        total_flows = len(df)
        malicious_mask = (preds != 0)
        malicious_total = int(np.sum(malicious_mask))
        ratio = (malicious_total / total_flows) * 100.0 if total_flows > 0 else 0.0

        self._update_metric_card(self.p3_m_total, "TOTAL FLOWS ANALYZED", str(total_flows))
        self._update_metric_card(self.p3_m_mal, "FLAGGED MALICIOUS", str(malicious_total), color="#ef4444" if malicious_total > 0 else "#22c55e")
        self._update_metric_card(self.p3_m_ratio, "MALICIOUS RATIO", f"{ratio:.2f}%", color="#ef4444" if ratio > 0 else "#22c55e")

        # Populate Table
        self.p3_table.setRowCount(min(100, total_flows)) # show first 100
        for row_i in range(min(100, total_flows)):
            pred_class = CLASSES[preds[row_i]]
            is_mal = "YES" if preds[row_i] != 0 else "NO"
            conf_val = f"{probs[row_i][preds[row_i]]*100:.2f}%"

            item_row = QTableWidgetItem(str(row_i + 1))
            item_class = QTableWidgetItem(pred_class)
            item_mal = QTableWidgetItem(is_mal)
            item_conf = QTableWidgetItem(conf_val)

            if preds[row_i] != 0:
                item_mal.setForeground(QColor("#ef4444"))
                item_class.setForeground(QColor("#fca5a5"))
            else:
                item_mal.setForeground(QColor("#22c55e"))

            self.p3_table.setItem(row_i, 0, item_row)
            self.p3_table.setItem(row_i, 1, item_class)
            self.p3_table.setItem(row_i, 2, item_mal)
            self.p3_table.setItem(row_i, 3, item_conf)

        # Plot Pie Chart
        threat_labels = [CLASSES[p] for p in preds]
        dist_df = pd.Series(threat_labels).value_counts().reset_index()
        dist_df.columns = ["Threat Category", "Count"]
        self.p3_canvas.plot_pie_chart(dist_df)

    def _on_csv_error(self, err_msg):
        QMessageBox.critical(self, "CSV Ingestion Error", f"Failed to process CSV file:\n{err_msg}")

    # ----------------- Page 4: Manual Sandbox -----------------
    def _build_page4_sandbox(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        controls_card = QFrame()
        controls_card.setObjectName("Card")
        grid = QGridLayout(controls_card)

        # Controls
        grid.addWidget(QLabel("Flow Duration (ms):"), 0, 0)
        self.sb_dur = QDoubleSpinBox()
        self.sb_dur.setRange(0.0, 10000.0)
        self.sb_dur.setValue(250.0)
        grid.addWidget(self.sb_dur, 0, 1)

        grid.addWidget(QLabel("Total Packets Sent:"), 0, 2)
        self.sb_pkts = QSpinBox()
        self.sb_pkts.setRange(1, 1000000)
        self.sb_pkts.setValue(100)
        grid.addWidget(self.sb_pkts, 0, 3)

        grid.addWidget(QLabel("Mean Packet Length (bytes):"), 1, 0)
        self.sb_len = QDoubleSpinBox()
        self.sb_len.setRange(0.0, 1500.0)
        self.sb_len.setValue(512.0)
        grid.addWidget(self.sb_len, 1, 1)

        grid.addWidget(QLabel("SYN Flags Count:"), 1, 2)
        self.sb_syn = QSpinBox()
        self.sb_syn.setRange(0, 100000)
        self.sb_syn.setValue(2)
        grid.addWidget(self.sb_syn, 1, 3)

        grid.addWidget(QLabel("Inter-Arrival Time (IAT Mean):"), 2, 0)
        self.sb_iat = QDoubleSpinBox()
        self.sb_iat.setRange(0.0, 1000.0)
        self.sb_iat.setValue(1.2)
        grid.addWidget(self.sb_iat, 2, 1)

        grid.addWidget(QLabel("RST Flags Count:"), 2, 2)
        self.sb_rst = QSpinBox()
        self.sb_rst.setRange(0, 100000)
        self.sb_rst.setValue(0)
        grid.addWidget(self.sb_rst, 2, 3)

        eval_btn = QPushButton("⚡ Evaluate Sandbox Parameters")
        eval_btn.clicked.connect(self._evaluate_sandbox)
        grid.addWidget(eval_btn, 3, 0, 1, 4)

        layout.addWidget(controls_card)

        # Verdict Cards
        sb_metrics_layout = QHBoxLayout()
        self.p4_m_class = self._create_metric_card("SANDBOX CLASSIFICATION", "---")
        self.p4_m_conf = self._create_metric_card("CONFIDENCE RATING", "---")
        self.p4_m_sev = self._create_metric_card("SEVERITY SCORE", "---")

        sb_metrics_layout.addWidget(self.p4_m_class)
        sb_metrics_layout.addWidget(self.p4_m_conf)
        sb_metrics_layout.addWidget(self.p4_m_sev)
        layout.addLayout(sb_metrics_layout)

        self.p4_canvas = DarkMatplotlibCanvas(width=6, height=3)
        layout.addWidget(self.p4_canvas, stretch=1)

        self.stacked_widget.addWidget(page)
        self._evaluate_sandbox()

    def _evaluate_sandbox(self):
        custom_v = np.zeros(NUM_FEATURES)
        custom_v[0] = self.sb_dur.value()
        custom_v[1] = self.sb_pkts.value()
        custom_v[3] = self.sb_len.value()
        custom_v[11] = self.sb_syn.value()
        custom_v[7] = self.sb_iat.value()
        custom_v[12] = self.sb_rst.value()

        idx, probs = self.evaluate_vector(custom_v)
        sev_label, sev_color = SEVERITY[idx]

        self._update_metric_card(self.p4_m_class, "SANDBOX CLASSIFICATION", CLASSES[idx])
        self._update_metric_card(self.p4_m_conf, "CONFIDENCE RATING", f"{probs[idx]*100:.2f}%")
        self._update_metric_card(self.p4_m_sev, "SEVERITY SCORE", sev_label, color=sev_color)

        self.p4_canvas.plot_bar_chart(CLASSES, probs, idx)

    # Helper UI Builders
    def _create_metric_card(self, title, val, color=None):
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("MetricTitle")
        lbl_val = QLabel(val)
        lbl_val.setObjectName("MetricValue")
        if color:
            lbl_val.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        return card

    def _update_metric_card(self, card, title, val, color=None):
        layout = card.layout()
        lbl_title = layout.itemAt(0).widget()
        lbl_val = layout.itemAt(1).widget()

        lbl_title.setText(title)
        lbl_val.setText(val)
        if color:
            lbl_val.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")
        else:
            lbl_val.setStyleSheet("color: #38bdf8; font-size: 20px; font-weight: bold;")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)

    window = NetShieldDesktopApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
