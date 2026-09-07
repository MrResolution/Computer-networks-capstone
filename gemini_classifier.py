import os
import json
import urllib.request
import urllib.error

# Default Gemini API Key provided by user
DEFAULT_GEMINI_KEY = "AIzaSyBmwAFGT7ds2A1ax4t1urK55yx6c7Zq9O0"

# Threat classes matching the neural network taxonomy
THREAT_CLASSES = [
    "Benign Normal Traffic",
    "DoS / DDoS Flood",
    "PortScan (Reconnaissance)",
    "Botnet Command & Control",
    "Infiltration & Exploit",
    "Web Attack (SQLi / XSS)",
    "Brute Force (SSH/FTP)",
    "Malware Data Exfiltration"
]


def _call_gemini(prompt_text, api_key=None, timeout=30):
    """Low-level helper: sends a prompt to Gemini and returns the raw text response."""
    key = api_key if (api_key and api_key.strip()) else os.environ.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)
    key = key.strip()

    if not key:
        return None, "Gemini API key is missing."

    models_to_try = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro"]

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}]
    }

    last_error = None
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", ""), None
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode()[:200]
            except Exception:
                pass
            last_error = f"HTTP {e.code}: {e.reason} — {err_body}"
        except Exception as e:
            last_error = str(e)

    return None, last_error


# ---------------------------------------------------------------------------
# 1. GEMINI DETECTION — Independent threat classification from raw telemetry
# ---------------------------------------------------------------------------

def gemini_detect_threat(flow_metrics, api_key=None):
    """
    Uses Gemini as an independent detection engine.
    Sends raw flow telemetry to Gemini and asks it to classify the traffic
    into one of the 8 threat categories, returning structured results.

    Returns dict:
        {
            "gemini_class": str,        # Predicted threat class name
            "gemini_class_idx": int,    # Index into THREAT_CLASSES (0-7)
            "gemini_confidence": str,   # Gemini's self-assessed confidence
            "gemini_severity": str,     # LOW / MEDIUM / HIGH / CRITICAL
            "gemini_reasoning": str,    # Short explanation of why
            "error": str or None        # Error message if call failed
        }
    """
    classes_list = "\n".join(f"  {i}. {c}" for i, c in enumerate(THREAT_CLASSES))

    prompt = f"""You are a network security intrusion detection system.
Analyze the following raw network flow telemetry metrics and classify the traffic.

[RAW FLOW TELEMETRY]
- Flow Duration: {flow_metrics.get('flow_duration', 0):.4f} ms
- Total Packets: {flow_metrics.get('total_packets', 0)}
- Total Bytes: {flow_metrics.get('total_bytes', 'N/A')}
- Mean Packet Length: {flow_metrics.get('mean_pkt_len', 0):.2f} bytes
- SYN Flag Count: {flow_metrics.get('syn_flags', 0)}
- RST Flag Count: {flow_metrics.get('rst_flags', 0)}
- FIN Flag Count: {flow_metrics.get('fin_flags', 0)}
- Inter-Arrival Time (IAT Mean): {flow_metrics.get('iat_mean', 0):.6f} ms
- Fwd Header Length: {flow_metrics.get('fwd_header_len', 'N/A')}

[CLASSIFICATION CATEGORIES]
{classes_list}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{"class_index": <0-7>, "class_name": "<exact name from list>", "confidence": "<HIGH/MEDIUM/LOW>", "severity": "<LOW/MEDIUM/HIGH/CRITICAL>", "reasoning": "<1-2 sentence explanation>"}}
"""

    text, error = _call_gemini(prompt, api_key=api_key, timeout=15)

    if error:
        return {
            "gemini_class": "Unknown",
            "gemini_class_idx": -1,
            "gemini_confidence": "N/A",
            "gemini_severity": "N/A",
            "gemini_reasoning": "",
            "error": error
        }

    # Parse JSON from Gemini response (strip markdown fences if present)
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(lines).strip()

    try:
        parsed = json.loads(clean)
        class_idx = int(parsed.get("class_index", -1))
        class_name = parsed.get("class_name", "Unknown")

        # Validate class_idx
        if 0 <= class_idx < len(THREAT_CLASSES):
            class_name = THREAT_CLASSES[class_idx]
        else:
            # Try to match by name
            for i, c in enumerate(THREAT_CLASSES):
                if c.lower() in class_name.lower() or class_name.lower() in c.lower():
                    class_idx = i
                    class_name = c
                    break

        return {
            "gemini_class": class_name,
            "gemini_class_idx": class_idx,
            "gemini_confidence": parsed.get("confidence", "N/A"),
            "gemini_severity": parsed.get("severity", "N/A"),
            "gemini_reasoning": parsed.get("reasoning", ""),
            "error": None
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        # Gemini didn't return clean JSON — extract what we can
        return {
            "gemini_class": "Unknown",
            "gemini_class_idx": -1,
            "gemini_confidence": "N/A",
            "gemini_severity": "N/A",
            "gemini_reasoning": text[:300] if text else "",
            "error": "Failed to parse Gemini JSON response"
        }


# ---------------------------------------------------------------------------
# 2. GEMINI THREAT REPORT — Deep forensic analysis after detection
# ---------------------------------------------------------------------------

def classify_threat_with_gemini(detected_class, confidence_pct, severity_label, flow_metrics=None, api_key=None):
    """
    Generates a detailed Threat Intelligence Report using Gemini,
    given a pre-existing detection result from the neural network.
    """
    metrics_str = ""
    if flow_metrics:
        metrics_str = f"""
Flow Telemetry Metrics:
- Flow Duration: {flow_metrics.get('flow_duration', 'N/A')} ms
- Total Packets: {flow_metrics.get('total_packets', 'N/A')}
- Mean Packet Length: {flow_metrics.get('mean_pkt_len', 'N/A')} bytes
- SYN Flag Count: {flow_metrics.get('syn_flags', 'N/A')}
- RST Flag Count: {flow_metrics.get('rst_flags', 'N/A')}
- Inter-Arrival Time (IAT Mean): {flow_metrics.get('iat_mean', 'N/A')} ms
"""

    prompt_text = f"""
You are NetShield AI's Senior Cyber Threat Intelligence Specialist.
Our neural network intrusion detection system has detected and flagged the following network traffic signature:

[DETECTION RESULTS]
- Primary Classification: {detected_class}
- Neural Model Confidence: {confidence_pct:.2f}%
- Threat Severity Level: {severity_label}
{metrics_str}

Please generate a professional, structured Threat Intelligence Report containing:
1. **Executive Threat Overview**: Concise explanation of what this attack mechanism is and how it behaves.
2. **Technical Vector & Signature Analysis**: Breakdown of the network behavior and packet indicators.
3. **Impact & Risk Assessment**: Business & infrastructure impact if unmitigated.
4. **Actionable Remediation & Firewall Rules**: 3-4 specific incident response steps (e.g. iptables/snort rule, rate limiting, IP quarantine).

Format the output clearly using Markdown headers and bullet points.
"""

    text, error = _call_gemini(prompt_text, api_key=api_key, timeout=30)
    if error:
        return f"❌ Gemini API Call Failed: {error}"
    return text


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Gemini Independent Detection")
    print("=" * 60)
    test_metrics = {
        'flow_duration': 45.0,
        'total_packets': 45000,
        'syn_flags': 42000,
        'mean_pkt_len': 64.0,
        'rst_flags': 10,
        'fin_flags': 0,
        'iat_mean': 0.0002,
        'total_bytes': 2880000,
        'fwd_header_len': 900000
    }
    detection = gemini_detect_threat(test_metrics)
    print(json.dumps(detection, indent=2))

    print("\n" + "=" * 60)
    print("TEST 2: Gemini Threat Report")
    print("=" * 60)
    report = classify_threat_with_gemini(
        detected_class=detection["gemini_class"],
        confidence_pct=99.85,
        severity_label="CRITICAL",
        flow_metrics=test_metrics
    )
    print(report[:500] + "...")
