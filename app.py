import json
import os
import secrets
import threading
import time

import fitz
import httpx
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

SERVER_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
SERVER_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

# Rate limiting: 3 analyses per IP per 24 hours (only enforced when server key is active)
RATE_LIMIT_WINDOW = 24 * 60 * 60   # seconds
RATE_LIMIT_MAX    = 3               # analyses allowed per window
RATE_LIMIT_FILE   = "rate_limit_store.json"
_rate_lock = threading.Lock()


def _load_rate_store() -> dict[str, list[float]]:
    if not os.path.exists(RATE_LIMIT_FILE):
        return {}
    try:
        with open(RATE_LIMIT_FILE, "r") as f:
            data = json.load(f)
            return {k: [float(t) for t in v] for k, v in data.items()}
    except Exception as e:
        print(f"Error loading rate limits file: {e}")
        return {}


def _save_rate_store(store: dict[str, list[float]]) -> None:
    try:
        with open(RATE_LIMIT_FILE, "w") as f:
            json.dump(store, f)
    except Exception as e:
        print(f"Error saving rate limits file: {e}")


_rate_store: dict[str, list[float]] = _load_rate_store()


def _get_client_ip() -> str:
    """Return the real client IP, respecting Render's X-Forwarded-For header."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _check_rate_limit(ip: str) -> tuple[bool, float]:
    """
    Returns (allowed, retry_after_seconds).
    Cleans up timestamps older than the window.
    Does NOT increment the count.
    """
    now = time.time()
    with _rate_lock:
        timestamps = _rate_store.get(ip, [])
        # Drop timestamps outside the window
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        _rate_store[ip] = timestamps
        _save_rate_store(_rate_store)
        if len(timestamps) >= RATE_LIMIT_MAX:
            retry_after = RATE_LIMIT_WINDOW - (now - timestamps[0])
            return False, retry_after
        return True, 0.0


def _record_rate_limit(ip: str) -> None:
    """Records a successful analysis request for the given IP."""
    now = time.time()
    with _rate_lock:
        timestamps = _rate_store.get(ip, [])
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        timestamps.append(now)
        _rate_store[ip] = timestamps
        _save_rate_store(_rate_store)


def _extract_json_block(text: str) -> str:
    text = text.strip()
    start = text.find('{')
    if start == -1:
        return text
    end = text.rfind('}')
    if end == -1:
        return text[start:]
    return text[start:end+1]


SYSTEM_PROMPT = """\
You are an expert medical document translator and clinical communicator. Your sole objective is to help patients understand their medical records, lab reports, and clinical summaries in clear, accessible, and jargon-free language.

### Core Ethical & Safety Constraints:
- NEVER diagnose any condition. Frame findings as "suggests," "indicates," or "is associated with," rather than "you have X."
- NEVER prescribe or recommend specific medications, dosages, or alternative treatments.
- NEVER contradict or override a physician's written instructions.
- ALWAYS emphasize that these explanations are educational and that they should consult their healthcare provider for any diagnostic or treatment decisions.
- Be clinically accurate and objective: do not offer false reassurance. If a value is abnormal or critical, explain its significance plainly and direct the patient to appropriate medical care.

### Extraction Instructions:
1. **Identify Metrics**: Scan the entire text and extract all lab values, vital signs (blood pressure, heart rate, temperature), clinical findings, or metrics.
2. **Value & Unit**: Record the exact value and unit (e.g., "mg/dL", "g/L", "mmHg") as written in the document.
3. **Reference Ranges**:
   - Extract the reference range provided in the document.
   - If no reference range is provided in the document, use standard clinical reference ranges for adults and label it clearly (e.g., "Standard range: 4.0 - 11.0").
4. **Determine Status**: Classify each metric status into one of:
   - `normal`: within the reference range.
   - `elevated`: above the upper limit of the reference range.
   - `low`: below the lower limit of the reference range.
   - `critical`: severely out of bounds, indicating potential acute issues.
5. **Determine Urgency**:
   - `routine`: No action needed beyond routine checkups.
   - `followup`: Discuss at the next scheduled doctor's visit.
   - `soon`: Book a non-urgent appointment within 1-2 weeks.
   - `urgent`: Contact a doctor or seek medical care within 24-48 hours.
6. **Plain Language Explanations (Conserve tokens)**:
   - For `normal` metrics, set "explanation" to "Within normal limits." (Do NOT write detailed explanations for normal values to save tokens).
   - Only for abnormal metrics (`elevated`, `low`, `critical`), write a 1-2 sentence explanation. Avoid jargon (explain "thrombocytopenia" as "a low platelet count, which are cells that help blood clot") and explain why it matters (e.g., "Creatinine is a waste product filtered by the kidneys; elevated levels can suggest the kidneys are working harder than usual.").

### Output Format:
Return ONLY valid JSON with no markdown formatting, no code block fences, and no conversational preamble. The output must strictly conform to the following schema:

{
  "summary": "A 2-3 sentence compassionate, plain-language overview of the overall document findings, summarizing the key takeaways and highlighting if there are any areas of concern.",
  "metrics": [
    {
      "name": "Full name of the metric (e.g., Thyroid Stimulating Hormone)",
      "value": "Value as written",
      "unit": "Unit of measurement (or empty string if none)",
      "normal_range": "Reference range (either from document or standard, e.g., '12.0 - 16.0 g/dL')",
      "status": "normal | elevated | low | critical",
      "urgency": "routine | followup | soon | urgent",
      "explanation": "Clear, patient-friendly explanation of what this test is and what the result means."
    }
  ],
  "overall_urgency": "routine | followup | soon | urgent",
  "recommended_actions": [
    "A list of 2-4 concrete, actionable next steps (e.g., 'Discuss the elevated TSH level with your doctor at your next visit', 'Keep a log of your daily blood pressure readings')."
  ]
}

### Urgency Definitions:
- `routine`: Normal results. No special action needed.
- `followup`: Mild abnormalities. Worth mentioning at the next scheduled visit.
- `soon`: Moderate abnormalities or patterns that should be evaluated within 2 weeks.
- `urgent`: Critical values or severe abnormalities that require reaching out to your doctor within 24-48 hours.
"""

HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mediq – AI Medical Document Explainer</title>
<meta name="description" content="Understand your health records in plain language with AI-powered analysis. Upload a PDF and get a clear, jargon-free explanation of your lab results.">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    /* Color Tokens - Light Mode */
    --bg: #F8FAFC;
    --surface: #FFFFFF;
    --border: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    
    --primary: #0D9488;
    --primary-hover: #0F766E;
    --primary-light: #F0FDFA;
    --accent: #0284C7;
    
    --danger: #EF4444;
    --danger-light: #FEF2F2;
    --warning: #F59E0B;
    --warning-light: #FEF3C7;
    --success: #10B981;
    --success-light: #ECFDF5;
    --info: #3B82F6;
    --info-light: #EFF6FF;

    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 12px;
    
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -4px rgba(0, 0, 0, 0.03);
    --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  [data-theme="dark"] {
    /* Color Tokens - Dark Mode */
    --bg: #0B0F17;
    --surface: #1E293B;
    --border: #334155;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    
    --primary: #14B8A6;
    --primary-hover: #2DD4BF;
    --primary-light: #132A29;
    --accent: #38BDF8;
    
    --danger: #F87171;
    --danger-light: #2C1B1B;
    --warning: #FBBF24;
    --warning-light: #2D2214;
    --success: #34D399;
    --success-light: #122D24;
    --info: #60A5FA;
    --info-light: #162235;
  }

  body {
    background: var(--bg);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
    transition: var(--transition);
  }

  header {
    background: var(--surface);
    padding: 1.25rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .logo-area {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-decoration: none;
    color: inherit;
  }
  .logo-icon {
    color: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .logo-title {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text-primary);
  }

  .nav-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .theme-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 0.5rem;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
  }
  .theme-btn:hover {
    color: var(--primary);
    border-color: var(--primary);
  }
  .theme-btn svg { width: 1.125rem; height: 1.125rem; }

  .theme-btn .sun-icon { display: block; }
  .theme-btn .moon-icon { display: none; }
  [data-theme="dark"] .theme-btn .sun-icon { display: none; }
  [data-theme="dark"] .theme-btn .moon-icon { display: block; }

  .container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 3rem 1.5rem;
    width: 100%;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    padding: 1.75rem;
    transition: var(--transition);
  }
  .card:hover {
    box-shadow: var(--shadow-md);
  }

  /* Buttons styling */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    font-family: inherit;
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.75rem 1.25rem;
    border-radius: var(--radius-md);
    border: none;
    cursor: pointer;
    transition: var(--transition);
    text-decoration: none;
  }
  .btn-primary {
    background: var(--primary);
    color: #ffffff;
  }
  .btn-primary:hover {
    background: var(--primary-hover);
  }
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: var(--text-muted);
  }
  .btn-outline {
    background: var(--surface);
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }
  .btn-outline:hover {
    background: var(--bg);
    color: var(--text-primary);
  }
  .btn-full { width: 100%; }

  /* Inputs styling */
  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
    margin-bottom: 1rem;
  }
  .input-group label {
    font-size: 0.825rem;
    font-weight: 500;
    color: var(--text-secondary);
  }
  .input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem 0.875rem;
    font-family: inherit;
    font-size: 0.9rem;
    color: var(--text-primary);
    outline: none;
    transition: var(--transition);
  }
  .input:focus {
    border-color: var(--primary);
  }

  .hint {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  /* Screens */
  #screen-landing {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    padding: 2rem 0;
    text-align: center;
    animation: fadeIn 0.4s ease-out;
  }
  #screen-landing h1 {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0.75rem;
    line-height: 1.15;
    color: var(--text-primary);
  }
  #screen-landing p.subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    margin-bottom: 2rem;
    max-width: 440px;
    line-height: 1.45;
  }
  .landing-form-card {
    width: 100%;
    max-width: 380px;
    text-align: left;
  }

  #screen-upload {
    display: none;
    flex-direction: column;
    flex: 1;
    animation: fadeIn 0.4s ease-out;
  }
  .upload-header {
    text-align: center;
    margin: 1rem 0 2rem;
  }
  .upload-header h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
  }
  .upload-header p {
    font-size: 0.95rem;
    color: var(--text-secondary);
  }

  .drop-zone {
    border: 1px dashed var(--border);
    border-radius: var(--radius-md);
    background: var(--surface);
    padding: 4rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: var(--transition);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--primary);
    background: var(--primary-light);
  }
  .drop-zone svg {
    color: var(--text-muted);
    transition: var(--transition);
  }
  .drop-zone:hover svg {
    color: var(--primary);
  }
  .drop-zone p.big {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  .drop-zone p.small {
    font-size: 0.85rem;
    color: var(--text-muted);
  }
  #file-input { display: none; }

  /* Processing */
  #screen-processing {
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    padding: 3rem 0;
    animation: fadeIn 0.3s ease-out;
  }
  .spinner-container {
    width: 48px;
    height: 48px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .spinner-ring {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  #processing-filename {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }
  .progress-steps {
    list-style: none;
    width: 100%;
    max-width: 280px;
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
    text-align: left;
    margin-top: 1rem;
  }
  .step {
    font-size: 0.85rem;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 0.625rem;
    transition: var(--transition);
  }
  .step-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--border);
    transition: var(--transition);
  }
  .step.active {
    color: var(--text-primary);
    font-weight: 500;
  }
  .step.active .step-dot {
    background: var(--primary);
  }
  .step.completed {
    color: var(--text-secondary);
  }
  .step.completed .step-dot {
    background: var(--success);
  }

  /* Results dashboard */
  #screen-results {
    display: none;
    flex-direction: column;
    flex: 1;
    gap: 1.5rem;
    animation: fadeIn 0.4s ease-out;
  }
  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    padding-bottom: 0.25rem;
  }
  .results-header h2 {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
  }
  .results-header .date {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5rem;
    align-items: start;
  }
  @media (min-width: 800px) {
    .dashboard-grid {
      grid-template-columns: 1.1fr 1.9fr;
    }
  }

  .section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--primary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.375rem;
  }
  
  /* Badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    font-weight: 500;
  }
  .badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }
  
  /* Urgency Classes */
  .urgency-routine { background: var(--success-light); color: var(--success); }
  .urgency-routine .badge-dot { background: var(--success); }
  .urgency-followup { background: var(--warning-light); color: var(--warning); }
  .urgency-followup .badge-dot { background: var(--warning); }
  .urgency-soon { background: var(--info-light); color: var(--info); }
  .urgency-soon .badge-dot { background: var(--info); }
  .urgency-urgent { background: var(--danger-light); color: var(--danger); }
  .urgency-urgent .badge-dot { background: var(--danger); }

  /* Table styling */
  .table-card {
    padding: 1.5rem;
  }
  .table-wrapper {
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
  }
  th {
    padding: 0.75rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  td {
    padding: 0.875rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
    font-size: 0.9rem;
  }
  
  /* Metric Row Interaction */
  .metric-row {
    cursor: pointer;
    transition: var(--transition);
  }
  .metric-row:hover {
    background: var(--bg);
  }
  .metric-row.expanded {
    background: var(--bg);
  }
  .metric-row.expanded td {
    border-bottom-color: transparent;
  }

  .metric-name-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .metric-name-text {
    font-weight: 500;
    color: var(--text-primary);
  }
  .chevron-icon {
    color: var(--text-muted);
    transition: var(--transition);
  }
  .metric-row:hover .chevron-icon {
    color: var(--primary);
  }

  /* Status Badges */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0.5rem;
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: capitalize;
  }
  .status-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
  }

  .status-normal { background: var(--success-light); color: var(--success); }
  .status-normal .status-dot { background: var(--success); }
  
  .status-elevated { background: var(--warning-light); color: var(--warning); }
  .status-elevated .status-dot { background: var(--warning); }
  
  .status-low { background: var(--info-light); color: var(--info); }
  .status-low .status-dot { background: var(--info); }
  
  .status-critical { background: var(--danger-light); color: var(--danger); }
  .status-critical .status-dot { background: var(--danger); }

  /* Explanations sub-panel */
  .detail-row {
    background: var(--bg);
  }
  .detail-row td {
    padding: 0 1rem 1rem;
  }
  .detail-content {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem;
    margin-top: -0.25rem;
    box-shadow: var(--shadow-sm);
    animation: fadeIn 0.2s ease-out;
  }
  .detail-content p {
    font-size: 0.85rem;
    line-height: 1.45;
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
  }
  .detail-content p:last-child { margin-bottom: 0; }
  .detail-content strong { color: var(--text-primary); }
  
  .detail-urgency {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }
  .urgency-label {
    display: inline-block;
    padding: 0.15rem 0.35rem;
    border-radius: var(--radius-sm);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  /* Action checklist */
  .actions-card {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .action-item {
    display: flex;
    align-items: flex-start;
    gap: 0.625rem;
    padding: 0.625rem 0.875rem;
    border-radius: var(--radius-md);
    background: var(--surface);
    border: 1px solid var(--border);
    font-size: 0.85rem;
    line-height: 1.4;
  }
  .action-checkbox {
    color: var(--primary);
    background: var(--primary-light);
    border-radius: 50%;
    padding: 0.1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.1rem;
    flex-shrink: 0;
  }

  .action-buttons-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
  }
  .action-buttons-row .btn {
    flex: 1;
    min-width: 180px;
  }

  /* Errors card */
  #screen-error {
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    padding: 3rem 0;
    animation: fadeIn 0.3s ease-out;
  }
  .error-card {
    max-width: 480px;
    border: 1px solid var(--danger);
    background: var(--danger-light);
    border-radius: var(--radius-md);
    padding: 1.5rem;
  }
  .error-card-inner {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
  }
  .error-card h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--danger);
  }
  .error-card p {
    color: var(--text-secondary);
    margin-top: 0.35rem;
    font-size: 0.85rem;
    line-height: 1.4;
  }
  #retry-btn { margin-top: 1.5rem; }

  /* Mobile style changes */
  .mobile-metrics { display: none; }

  footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    padding: 1.5rem;
    margin-top: auto;
    line-height: 1.6;
  }

  @media (max-width: 600px) {
    .container { padding: 1.5rem 1rem; }
    .results-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
    .table-wrapper { display: none; }
    .mobile-metrics { display: flex; flex-direction: column; gap: 0.75rem; }
    
    .metric-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 1rem;
      cursor: pointer;
      transition: var(--transition);
    }
    .metric-card.expanded {
      background: var(--bg);
    }
    .metric-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    .metric-card-title {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .metric-card .metric-name {
      font-weight: 600;
      color: var(--text-primary);
    }
    .metric-val-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      margin-bottom: 0.25rem;
    }
    .metric-val-row .label {
      color: var(--text-secondary);
    }
    .metric-val-row .val-text {
      font-weight: 500;
    }
    .action-buttons-row { flex-direction: column; }
    .action-buttons-row .btn { width: 100%; }
  }

  @media (min-width: 601px) {
    .table-wrapper { display: block; }
    .mobile-metrics { display: none; }
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Self-hosting Guide styles */
  .guide-container {
    max-width: 680px;
    width: 100%;
    margin: 1.5rem auto 0;
    text-align: left;
    animation: fadeIn 0.4s ease-out;
  }
  .guide-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .guide-steps {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .guide-step {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    transition: var(--transition);
  }
  .guide-step:hover {
    border-color: var(--primary);
  }
  .guide-step-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
  }
  .guide-step-number {
    background: var(--primary-light);
    color: var(--primary);
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
  }
  .guide-step-body {
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }
  .guide-step-body a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
  }
  .guide-step-body a:hover {
    text-decoration: underline;
  }
  .code-block-container {
    margin-top: 0.75rem;
    position: relative;
    border-radius: var(--radius-md);
    overflow: hidden;
  }
  .code-block-header {
    background: #1E293B;
    color: #94A3B8;
    font-size: 0.75rem;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #334155;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 500;
  }
  .code-block {
    background: #0B0F17;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    padding: 0.75rem 1rem;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.5;
  }
  .guide-tab-buttons {
    display: flex;
    gap: 0.375rem;
    margin-top: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.25rem;
  }
  .guide-tab-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    padding: 0.375rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    border-radius: var(--radius-sm);
    transition: var(--transition);
  }
  .guide-tab-btn:hover {
    background: var(--bg);
    color: var(--text-primary);
  }
  .guide-tab-btn.active {
    background: var(--primary-light);
    color: var(--primary);
    font-weight: 600;
  }
</style>
</head>
<body>

<header>
  <a href="/" class="logo-area" onclick="event.preventDefault(); showLanding()">
    <div class="logo-icon">
      <svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
    </div>
    <span class="logo-title">Mediq</span>
  </a>
  <div class="nav-controls">
    <button class="theme-btn" onclick="toggleTheme()" aria-label="Toggle Theme">
      <svg class="sun-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707M12 8a4 4 0 100 8 4 4 0 000-8z"/>
      </svg>
      <svg class="moon-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
      </svg>
    </button>
  </div>
</header>

<div class="container">

  <div id="screen-landing">
    <h1>Understand Your Health</h1>
    <p class="subtitle">Decipher complex medical records and lab results in plain, patient-friendly language.</p>
    <div class="card landing-form-card">
      <form id="landing-form" onsubmit="submitApiKey(event)">
        <div class="input-group">
          <label for="provider-select">API Provider</label>
          <select id="provider-select" class="input" onchange="handleProviderChange()">
            <option value="openrouter" selected>OpenRouter</option>
            <option value="groq">Groq</option>
          </select>
        </div>
        <div class="input-group" id="api-key-group">
          <label id="api-key-label" for="api-key-input">OpenRouter API Key</label>
          <input id="api-key-input" type="password" class="input" autocomplete="off" spellcheck="false" placeholder="sk-or-...">
          <span class="hint">Your key is stored securely in your browser's session storage.</span>
        </div>
        <button id="continue-btn" type="submit" class="btn btn-primary btn-full" disabled>
          Continue
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
          </svg>
        </button>
      </form>
    </div>
  </div>

  <div id="screen-upload">
    <div class="upload-header">
      <h1>Analyze Medical PDF</h1>
      <p>Upload a text-based medical report to begin the translation.</p>
    </div>
    <div class="card" style="max-width: 600px; margin: 0 auto; width: 100%;">
      <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <svg width="56" height="56" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
        </svg>
        <p class="big">Drag & drop your PDF file here</p>
        <p class="small">or click to browse from device (max 10MB)</p>
        <input id="file-input" type="file" accept="application/pdf" onchange="handleFileSelect(event)">
      </div>
      
      <div style="display: flex; justify-content: space-between; align-items:center; margin-top: 1.5rem; font-size: 0.85rem; flex-wrap:wrap; gap:0.5rem;">
        <span id="quota-badge" style="display:none; font-size:0.78rem; padding:0.2rem 0.6rem; border-radius:9999px; background:var(--success-light); color:var(--success); font-weight:600;"></span>
        <a id="change-key-link" href="#" style="color: var(--text-muted); text-decoration:none;" onclick="event.preventDefault(); showLanding()">Change API key / Model</a>
      </div>
    </div>
  </div>

  <div id="screen-processing">
    <div class="spinner-container">
      <div class="spinner-ring"></div>
    </div>
    <p id="processing-filename"></p>
    <ul class="progress-steps">
      <li id="step-1" class="step"><span class="step-dot"></span>Reading medical document...</li>
      <li id="step-2" class="step"><span class="step-dot"></span>Extracting health metrics...</li>
      <li id="step-3" class="step"><span class="step-dot"></span>Analyzing with AI model...</li>
      <li id="step-4" class="step"><span class="step-dot"></span>Generating plain-language guide...</li>
    </ul>
  </div>

  <div id="screen-results">
    <div class="results-header">
      <div>
        <h2 id="results-filename"></h2>
        <p class="date" id="results-date"></p>
      </div>
      <div id="urgency-badge" class="badge"></div>
    </div>

    <div class="dashboard-grid">
      <!-- Left Panel: Summary & Actions -->
      <div style="display: flex; flex-direction: column; gap: 1.5rem;">
        <div class="card">
          <p class="section-label">
            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
            Summary Overview
          </p>
          <p id="results-summary" style="line-height:1.7; font-size: 0.95rem; color: var(--text-secondary);"></p>
        </div>

        <div class="card">
          <p class="section-label">
            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/></svg>
            Recommended Next Steps
          </p>
          <div id="actions-list" class="actions-card"></div>
        </div>
      </div>

      <!-- Right Panel: Results table -->
      <div class="card table-card">
        <p class="section-label">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z"/></svg>
          Extracted Lab Results
        </p>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-top:-0.5rem; margin-bottom:1.5rem;">Click on any result row to view a patient-friendly explanation.</p>
        
        <div class="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Metric Name</th>
                <th>Result</th>
                <th>Reference Range</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="metrics-table-body"></tbody>
          </table>
        </div>
        
        <div class="mobile-metrics" id="metrics-mobile"></div>
      </div>
    </div>

    <div class="action-buttons-row">
      <button class="btn btn-outline" onclick="showUpload()">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 7.89H18v3"/>
        </svg>
        Analyze Another Document
      </button>
      <button class="btn btn-primary" onclick="downloadSummary()">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
        </svg>
        Download Summary File
      </button>
    </div>
  </div>

  <div id="screen-error">
    <div class="card error-card" style="width: 100%; max-width: 680px; margin-bottom: 1.5rem;">
      <div class="error-card-inner">
        <svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="flex-shrink:0;color:var(--danger)">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <div>
          <h3 id="error-title">Analysis Failed</h3>
          <p id="error-detail">An unknown error occurred during parsing.</p>
        </div>
      </div>
    </div>

    <div style="display: flex; gap: 1rem; justify-content: center; width: 100%; margin-bottom: 1rem;">
      <button id="retry-btn" class="btn btn-primary" onclick="showUpload()">
        Try Uploading Again
      </button>
      <button class="btn btn-outline" onclick="showLanding()">
        Change API key / Provider
      </button>
    </div>

    <!-- Self-host instructions block -->
    <div id="self-host-instructions" class="guide-container" style="display:none;">
      <div class="guide-title">
        <svg width="20" height="20" fill="none" stroke="var(--primary)" stroke-width="2.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
        </svg>
        Mediq Local Setup Guide
      </div>
      
      <div class="guide-steps">
        <!-- Step 1 -->
        <div class="guide-step">
          <div class="guide-step-header">
            <span class="guide-step-number">1</span>
            <span>Obtain API Keys</span>
          </div>
          <div class="guide-step-body">
            To power the AI analysis, you need an API key from one of the supported providers:
            <ul style="margin-top: 0.5rem; margin-left: 1.25rem; display: flex; flex-direction: column; gap: 0.35rem;">
              <li>
                <strong>OpenRouter:</strong> Go to <a href="https://openrouter.ai/" target="_blank" rel="noopener">openrouter.ai</a>, sign up, and generate an API key in the <strong>API Keys</strong> section.
              </li>
              <li>
                <strong>Groq Cloud:</strong> Go to <a href="https://console.groq.com/" target="_blank" rel="noopener">console.groq.com</a>, sign up, and create an API key in the <strong>API Keys</strong> section.
              </li>
            </ul>
          </div>
        </div>

        <!-- Step 2 -->
        <div class="guide-step">
          <div class="guide-step-header">
            <span class="guide-step-number">2</span>
            <span>Prepare Environment</span>
          </div>
          <div class="guide-step-body">
            Make sure you have **Python 3.10+** and **Git** installed on your system. 
            Open your terminal (or command prompt) to clone the project and configure dependencies:
          </div>
        </div>

        <!-- Step 3 -->
        <div class="guide-step">
          <div class="guide-step-header">
            <span class="guide-step-number">3</span>
            <span>Installation & Running the App</span>
          </div>
          <div class="guide-step-body">
            Select your Operating System, copy the commands below, and run them in your terminal:
            
            <div class="guide-tab-buttons">
              <button type="button" class="guide-tab-btn active" onclick="switchOsTab('linux', this)">Linux / macOS</button>
              <button type="button" class="guide-tab-btn" onclick="switchOsTab('win-cmd', this)">Windows (CMD)</button>
              <button type="button" class="guide-tab-btn" onclick="switchOsTab('win-ps', this)">Windows (PowerShell)</button>
            </div>
            
            <div id="os-linux" class="os-tab-content" style="margin-top: 0.5rem;">
              <div class="code-block-container">
                <div class="code-block-header">
                  <span>Terminal</span>
                  <button type="button" onclick="copyCode('code-linux', this)" style="background:none;border:none;color:inherit;cursor:pointer;font-size:0.7rem;display:flex;align-items:center;gap:3px;">
                    <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                    Copy
                  </button>
                </div>
                <pre class="code-block" id="code-linux">git clone https://github.com/hiri-dev/Mediq.git
cd Mediq
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_openrouter_api_key"
export GROQ_API_KEY="your_groq_api_key"
python app.py</pre>
              </div>
            </div>
            
            <div id="os-win-cmd" class="os-tab-content" style="display:none; margin-top: 0.5rem;">
              <div class="code-block-container">
                <div class="code-block-header">
                  <span>Command Prompt (CMD)</span>
                  <button type="button" onclick="copyCode('code-win-cmd', this)" style="background:none;border:none;color:inherit;cursor:pointer;font-size:0.7rem;display:flex;align-items:center;gap:3px;">
                    <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                    Copy
                  </button>
                </div>
                <pre class="code-block" id="code-win-cmd">git clone https://github.com/hiri-dev/Mediq.git
cd Mediq
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set OPENROUTER_API_KEY=your_openrouter_api_key
set GROQ_API_KEY=your_groq_api_key
python app.py</pre>
              </div>
            </div>
            
            <div id="os-win-ps" class="os-tab-content" style="display:none; margin-top: 0.5rem;">
              <div class="code-block-container">
                <div class="code-block-header">
                  <span>PowerShell</span>
                  <button type="button" onclick="copyCode('code-win-ps', this)" style="background:none;border:none;color:inherit;cursor:pointer;font-size:0.7rem;display:flex;align-items:center;gap:3px;">
                    <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                    Copy
                  </button>
                </div>
                <pre class="code-block" id="code-win-ps">git clone https://github.com/hiri-dev/Mediq.git
cd Mediq
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENROUTER_API_KEY="your_openrouter_api_key"
$env:GROQ_API_KEY="your_groq_api_key"
python app.py</pre>
              </div>
            </div>
            
            <p style="margin-top: 0.75rem; font-size: 0.85rem; color: var(--text-muted);">
              Once running, open <a href="http://localhost:5000" target="_blank" rel="noopener">http://localhost:5000</a> in your browser. The application will run locally and use your provided keys directly.
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>

<footer>
  ⚕ Mediq is an educational aid powered by AI, and does not provide medical advice, diagnosis, or treatment recommendations.<br>
  Always verify report details and reference ranges with your clinical provider before initiating or altering any medical plan.<br>
  If you are experiencing severe symptoms or a medical emergency, call your local emergency services instantly.
</footer>

<script>
let serverOpenRouterKeyConfigured = false;
let serverGroqKeyConfigured = false;

let selectedProvider = sessionStorage.getItem('mediq_provider') || 'openrouter';
let openRouterKey = sessionStorage.getItem('mediq_openrouter_key') || '';
let groqKey = sessionStorage.getItem('mediq_groq_key') || '';
let selectedModel = sessionStorage.getItem('mediq_model') || 'openai/gpt-oss-20b';
let currentResults = null;
let currentFilename = '';

Object.defineProperty(window, 'serverKeyConfigured', {
  get: () => selectedProvider === 'openrouter' ? serverOpenRouterKeyConfigured : serverGroqKeyConfigured
});

const URGENCY = {
  routine:  { bg: 'var(--success-light)', text: 'var(--success)', dot: 'var(--success)', label: 'Normal - Routine Follow-up' },
  followup: { bg: 'var(--warning-light)', text: 'var(--warning)', dot: 'var(--warning)', label: 'Mild - Mention at next visit' },
  soon:     { bg: 'var(--info-light)', text: 'var(--info)', dot: 'var(--info)', label: 'Moderate - Book appointment soon' },
  urgent:   { bg: 'var(--danger-light)', text: 'var(--danger)', dot: 'var(--danger)', label: 'High Urgency - Contact doctor within 24-48h' },
};

const STATUS_COLORS = {
  normal: '#10B981', elevated: '#F59E0B', low: '#3B82F6', critical: '#EF4444',
};

// Initialize Theme
function getTheme() {
  return localStorage.getItem('mediq_theme') || 'light';
}
document.documentElement.setAttribute('data-theme', getTheme());

function toggleTheme() {
  const current = getTheme();
  const next = current === 'light' ? 'dark' : 'light';
  localStorage.setItem('mediq_theme', next);
  document.documentElement.setAttribute('data-theme', next);
}

function show(id) {
  ['landing','upload','processing','results','error'].forEach(s => {
    const el = document.getElementById('screen-' + s);
    el.style.display = (s === id) ? 'flex' : 'none';
  });
}

function showLanding() {
  resetErrorScreen();
  const hasServerKey = (selectedProvider === 'openrouter' ? serverOpenRouterKeyConfigured : serverGroqKeyConfigured);
  if (hasServerKey) {
    show('landing');
    return;
  }
  
  if (selectedProvider === 'openrouter') {
    sessionStorage.removeItem('mediq_openrouter_key');
    openRouterKey = '';
  } else {
    sessionStorage.removeItem('mediq_groq_key');
    groqKey = '';
  }
  show('landing');
}

function showUpload() {
  resetErrorScreen();
  // Hide the "Change API key" link when the server manages the key
  const changeLink = document.getElementById('change-key-link');
  if (changeLink) changeLink.style.display = serverKeyConfigured ? 'none' : '';
  show('upload');
}

const keyInput = document.getElementById('api-key-input');
const continueBtn = document.getElementById('continue-btn');

keyInput.addEventListener('input', () => {
  const prov = document.getElementById('provider-select').value;
  const hasServerKey = (prov === 'openrouter' ? serverOpenRouterKeyConfigured : serverGroqKeyConfigured);
  if (hasServerKey) {
    continueBtn.disabled = false;
  } else {
    continueBtn.disabled = !keyInput.value.trim();
  }
});

function submitApiKey(e) {
  e.preventDefault();
  const k = keyInput.value.trim();
  const prov = document.getElementById('provider-select').value;
  
  selectedProvider = prov;
  sessionStorage.setItem('mediq_provider', prov);
  
  const hasServerKey = (prov === 'openrouter' ? serverOpenRouterKeyConfigured : serverGroqKeyConfigured);
  if (!hasServerKey) {
    if (!k) return;
    if (prov === 'openrouter') {
      openRouterKey = k;
      sessionStorage.setItem('mediq_openrouter_key', k);
    } else {
      groqKey = k;
      sessionStorage.setItem('mediq_groq_key', k);
    }
  }
  showUpload();
}

const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) processFile(f);
});

function handleFileSelect(e) {
  const f = e.target.files[0];
  if (f) processFile(f);
}

// Progress Steps Animator
let msgTimer;
let currentStep = 1;
function startProgressAnimation() {
  currentStep = 1;
  updateProgressSteps();
  msgTimer = setInterval(() => {
    if (currentStep < 4) {
      currentStep++;
      updateProgressSteps();
    }
  }, 2200);
}

function updateProgressSteps() {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById('step-' + i);
    if (!el) continue;
    if (i < currentStep) {
      el.className = 'step completed';
    } else if (i === currentStep) {
      el.className = 'step active';
    } else {
      el.className = 'step';
    }
  }
}

function stopProgressAnimation() {
  clearInterval(msgTimer);
}

async function processFile(file) {
  const limitUntil = localStorage.getItem('mediq_limit_until');
  if (limitUntil) {
    const remainingMs = parseInt(limitUntil) - Date.now();
    if (remainingMs > 0) {
      showRateLimit(Math.ceil(remainingMs / 1000));
      return;
    } else {
      localStorage.removeItem('mediq_limit_until');
    }
  }

  if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
    showError('Invalid file type', 'Please upload a PDF file.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File too large', 'Max file size is 10 MB.');
    return;
  }

  currentFilename = file.name;
  document.getElementById('processing-filename').textContent = file.name;
  show('processing');
  startProgressAnimation();
  resetErrorScreen();

  // Determine which providers to try
  const providersToTry = [selectedProvider];
  const fallback = (selectedProvider === 'openrouter') ? 'groq' : 'openrouter';
  const isFallbackConfigured = (fallback === 'openrouter')
    ? (serverOpenRouterKeyConfigured || !!openRouterKey)
    : (serverGroqKeyConfigured || !!groqKey);

  if (isFallbackConfigured) {
    providersToTry.push(fallback);
  }

  let lastError = null;

  for (let i = 0; i < providersToTry.length; i++) {
    const provider = providersToTry[i];
    const isProvServerConfigured = (provider === 'openrouter') ? serverOpenRouterKeyConfigured : serverGroqKeyConfigured;
    const key = (provider === 'openrouter') ? openRouterKey : groqKey;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('provider', provider);
    if (!isProvServerConfigured) {
      formData.append('api_key', key);
    }
    formData.append('model', selectedModel);

    try {
      const resp = await fetch('/analyze', { method: 'POST', body: formData });
      
      let data;
      const contentType = resp.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        data = await resp.json();
      } else {
        throw new Error(`The server returned an unexpected response format (${resp.status}).`);
      }

      if (!resp.ok) {
        if (resp.status === 401 && !isProvServerConfigured) {
          // If api key fails authorization and it is user-provided, redirect to landing
          stopProgressAnimation();
          showLanding();
          return;
        }
        if (resp.status === 429) {
          stopProgressAnimation();
          const retryAfter = data.retry_after || 86400;
          showRateLimit(retryAfter);
          return;
        }
        throw new Error(data.detail || data.error || 'Something went wrong.');
      }

      // Success
      stopProgressAnimation();
      if (provider !== selectedProvider) {
        selectedProvider = provider;
        sessionStorage.setItem('mediq_provider', provider);
        const provSelect = document.getElementById('provider-select');
        if (provSelect) provSelect.value = provider;
        handleProviderChange();
      }
      currentResults = data;
      renderResults(data, file.name);
      show('results');
      return;
    } catch (err) {
      console.warn(`Attempt with provider "${provider}" failed:`, err);
      lastError = err;
      // Continue loop to fallback provider if available
    }
  }

  // If we reach here, all attempted providers failed
  stopProgressAnimation();
  showError('Analysis Failed', lastError ? lastError.message : 'All available AI providers failed to process the request.');
  showSelfHostInstructions();
}

function renderResults(r, filename) {
  const trunc = s => s.length > 35 ? s.slice(0, 35) + '…' : s;
  document.getElementById('results-filename').textContent = trunc(filename);
  document.getElementById('results-date').textContent = new Date().toLocaleString();
  document.getElementById('results-summary').textContent = r.summary || '';

  const u = URGENCY[r.overall_urgency] || URGENCY.routine;
  const badge = document.getElementById('urgency-badge');
  badge.style.background = u.bg;
  badge.style.color = u.text;
  badge.innerHTML = `<span class="badge-dot" style="background:${u.dot}"></span>${u.label}`;

  const tbody = document.getElementById('metrics-table-body');
  tbody.innerHTML = '';
  const mobile = document.getElementById('metrics-mobile');
  mobile.innerHTML = '';

  (r.metrics || []).forEach((m, idx) => {
    // Desktop row rendering
    const trMetric = document.createElement('tr');
    trMetric.className = 'metric-row';
    trMetric.dataset.index = idx;
    trMetric.onclick = () => toggleRowDetail(idx);
    
    trMetric.innerHTML = `
      <td>
        <div class="metric-name-container">
          <span class="metric-name-text">${esc(m.name)}</span>
          <svg class="chevron-icon" id="chevron-${idx}" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/>
          </svg>
        </div>
      </td>
      <td class="mono">${esc(m.value)} ${esc(m.unit)}</td>
      <td style="color:var(--text-secondary)">${esc(m.normal_range)}</td>
      <td>
        <span class="status-badge status-${m.status}">
          <span class="status-dot"></span>
          ${esc(m.status)}
        </span>
      </td>
    `;
    
    const trDetail = document.createElement('tr');
    trDetail.id = `detail-row-${idx}`;
    trDetail.className = 'detail-row';
    trDetail.style.display = 'none';
    trDetail.innerHTML = `
      <td colspan="4">
        <div class="detail-content">
          <p><strong>Plain Explanations:</strong> ${esc(m.explanation)}</p>
          <div class="detail-urgency">
            <strong>Action Schedule:</strong>
            <span class="urgency-label urgency-${m.urgency}">${esc(m.urgency)}</span>
          </div>
        </div>
      </td>
    `;
    
    tbody.appendChild(trMetric);
    tbody.appendChild(trDetail);

    // Mobile list card rendering
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.onclick = () => toggleMobileCardDetail(card);
    card.innerHTML = `
      <div class="metric-card-header">
        <div class="metric-card-title">
          <span class="metric-name">${esc(m.name)}</span>
          <span class="status-badge status-${m.status}">
            <span class="status-dot"></span>
            ${esc(m.status)}
          </span>
        </div>
        <svg class="chevron-icon" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/>
        </svg>
      </div>
      <div class="metric-val-row">
        <span class="label">Result:</span>
        <span class="mono val-text">${esc(m.value)} ${esc(m.unit)}</span>
      </div>
      <div class="metric-val-row">
        <span class="label">Range:</span>
        <span class="muted">${esc(m.normal_range)}</span>
      </div>
      <div class="mobile-detail-content" style="display:none; margin-top:0.75rem; border-top:1px solid var(--border); padding-top:0.75rem;">
        <p style="font-size:0.85rem; line-height:1.4; color:var(--text-secondary)"><strong>Explanation:</strong> ${esc(m.explanation)}</p>
        <p style="margin-top:0.5rem; font-size:0.85rem; display:flex; align-items:center; gap:0.5rem;">
          <strong>Timeline:</strong>
          <span class="urgency-label urgency-${m.urgency}">${esc(m.urgency)}</span>
        </p>
      </div>
    `;
    mobile.appendChild(card);
  });

  const actionsList = document.getElementById('actions-list');
  actionsList.innerHTML = '';
  (r.recommended_actions || []).forEach((a, i) => {
    const div = document.createElement('div');
    div.className = 'action-item';
    div.innerHTML = `
      <span class="action-checkbox">
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="3" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
        </svg>
      </span>
      <span>${esc(a)}</span>
    `;
    actionsList.appendChild(div);
  });
}

function toggleRowDetail(idx) {
  const detailRow = document.getElementById(`detail-row-${idx}`);
  const chevron = document.getElementById(`chevron-${idx}`);
  if (!detailRow) return;
  
  if (detailRow.style.display === 'none') {
    detailRow.style.display = 'table-row';
    chevron.style.transform = 'rotate(180deg)';
    chevron.style.color = 'var(--primary)';
  } else {
    detailRow.style.display = 'none';
    chevron.style.transform = 'rotate(0deg)';
    chevron.style.color = 'var(--text-muted)';
  }
}

function toggleMobileCardDetail(card) {
  const detail = card.querySelector('.mobile-detail-content');
  const chevron = card.querySelector('.chevron-icon');
  
  if (detail.style.display === 'none') {
    detail.style.display = 'block';
    card.classList.add('expanded');
    chevron.style.transform = 'rotate(180deg)';
    chevron.style.color = 'var(--primary)';
  } else {
    detail.style.display = 'none';
    card.classList.remove('expanded');
    chevron.style.transform = 'rotate(0deg)';
    chevron.style.color = 'var(--text-muted)';
  }
}

function downloadSummary() {
  if (!currentResults) return;
  const r = currentResults;
  const u = URGENCY[r.overall_urgency] || URGENCY.routine;
  let txt = `DOCUMENT SUMMARY: ${currentFilename}\n\n`;
  txt += `URGENCY: ${u.label}\n\n`;
  txt += `OVERVIEW\n${r.summary}\n\n`;
  txt += `YOUR RESULTS\n`;
  (r.metrics || []).forEach(m => {
    txt += `- ${m.name}: ${m.value} ${m.unit} (Normal Range: ${m.normal_range}) — Status: ${m.status}\n`;
    txt += `  Explanation: ${m.explanation}\n`;
  });
  txt += `\nWHAT TO DO\n`;
  (r.recommended_actions || []).forEach((a, i) => { txt += `${i + 1}. ${a}\n`; });

  const blob = new Blob([txt], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'summary-' + currentFilename + '.txt';
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

function showError(title, detail) {
  document.getElementById('error-title').textContent = title;
  document.getElementById('error-detail').textContent = detail;
  show('error');
}

function showSelfHostInstructions() {
  const guide = document.getElementById('self-host-instructions');
  if (guide) guide.style.display = 'block';
}

function resetErrorScreen() {
  const guide = document.getElementById('self-host-instructions');
  if (guide) guide.style.display = 'none';
  const retryBtn = document.getElementById('retry-btn');
  if (retryBtn) retryBtn.style.display = '';
}

function switchOsTab(os, btn) {
  const parent = btn.parentElement;
  const buttons = parent.querySelectorAll('.guide-tab-btn');
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  
  const stepBody = parent.parentElement;
  const contents = stepBody.querySelectorAll('.os-tab-content');
  contents.forEach(c => c.style.display = 'none');
  
  if (os === 'linux') {
    stepBody.querySelector('#os-linux').style.display = 'block';
  } else if (os === 'win-cmd') {
    stepBody.querySelector('#os-win-cmd').style.display = 'block';
  } else if (os === 'win-ps') {
    stepBody.querySelector('#os-win-ps').style.display = 'block';
  }
}

function copyCode(elementId, btn) {
  const text = document.getElementById(elementId).innerText;
  navigator.clipboard.writeText(text).then(() => {
    const originalHTML = btn.innerHTML;
    btn.innerHTML = `<svg width="12" height="12" fill="none" stroke="#10B981" stroke-width="3" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg> <span style="color:#10B981">Copied!</span>`;
    setTimeout(() => {
      btn.innerHTML = originalHTML;
    }, 2000);
  }).catch(err => {
    console.error("Failed to copy code block: ", err);
  });
}

function showRateLimit(retryAfterSeconds) {
  // Store the block end timestamp in localStorage
  const limitUntil = Date.now() + retryAfterSeconds * 1000;
  localStorage.setItem('mediq_limit_until', limitUntil);

  // Show the error screen with a live countdown
  const fmt = (s) => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    return `${h}h ${m}m ${sec}s`;
  };
  let remaining = retryAfterSeconds;
  document.getElementById('error-title').textContent = '\u23F0 Daily Limit Reached';
  const updateDetail = () => {
    document.getElementById('error-detail').textContent =
      `You\'ve reached your daily free analysis limit. Resets in ${fmt(remaining)}.`;
  };
  updateDetail();
  show('error');
  // Hide retry button
  document.getElementById('retry-btn').style.display = 'none';
  
  if (window.rateLimitInterval) clearInterval(window.rateLimitInterval);
  window.rateLimitInterval = setInterval(() => {
    remaining--;
    if (remaining <= 0) {
      clearInterval(window.rateLimitInterval);
      localStorage.removeItem('mediq_limit_until');
      document.getElementById('retry-btn').style.display = '';
      document.getElementById('error-detail').textContent = 'Your limit has reset! You can analyze a new document.';
    } else {
      updateDetail();
    }
  }, 1000);
}

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function handleProviderChange() {
  const provSelect = document.getElementById('provider-select');
  const prov = provSelect.value;
  selectedProvider = prov;
  
  // Update API key label and placeholder
  const label = document.getElementById('api-key-label');
  const input = document.getElementById('api-key-input');
  
  if (prov === 'openrouter') {
    label.textContent = 'OpenRouter API Key';
    input.placeholder = 'sk-or-...';
    input.value = openRouterKey;
  } else {
    label.textContent = 'Groq API Key';
    input.placeholder = 'gsk_...';
    input.value = groqKey;
  }
  
  // Show/hide API key group based on server config
  const hasServerKey = (prov === 'openrouter' ? serverOpenRouterKeyConfigured : serverGroqKeyConfigured);
  const keyGroup = document.getElementById('api-key-group');
  if (keyGroup) {
    keyGroup.style.display = hasServerKey ? 'none' : 'block';
  }
  
  // Update continue button
  const continueBtn = document.getElementById('continue-btn');
  if (hasServerKey) {
    continueBtn.disabled = false;
  } else {
    continueBtn.disabled = !input.value.trim();
  }
}

async function init() {
  // Check client-side localStorage rate limit block
  const limitUntil = localStorage.getItem('mediq_limit_until');
  if (limitUntil) {
    const remainingMs = parseInt(limitUntil) - Date.now();
    if (remainingMs > 0) {
      showRateLimit(Math.ceil(remainingMs / 1000));
      return;
    } else {
      localStorage.removeItem('mediq_limit_until');
    }
  }

  try {
    const res = await fetch('/config');
    const cfg = await res.json();
    serverOpenRouterKeyConfigured = cfg.openrouter_key_configured === true;
    serverGroqKeyConfigured = cfg.groq_key_configured === true;
  } catch (_) {
    serverOpenRouterKeyConfigured = false;
    serverGroqKeyConfigured = false;
  }

  // Populate provider selector
  const provSelect = document.getElementById('provider-select');
  if (provSelect) {
    provSelect.value = selectedProvider;
  }
  
  handleProviderChange();
  
  const hasServerKey = (selectedProvider === 'openrouter' ? serverOpenRouterKeyConfigured : serverGroqKeyConfigured);
  const activeKey = getActiveApiKey();
  
  if (hasServerKey || activeKey) {
    if (hasServerKey) {
      try {
        const qRes = await fetch('/quota');
        const q = await qRes.json();
        if (q.limited) {
          showRateLimit(q.retry_after);
          return;
        }
        const badge = document.getElementById('quota-badge');
        if (badge) {
          badge.textContent = `\u2713 ${q.remaining} of ${q.max} free analyses left today`;
          badge.style.display = 'inline-block';
        }
      } catch (_) {}
    }
    showUpload();
  } else {
    show('landing');
  }
}

function getActiveApiKey() {
  return selectedProvider === 'openrouter' ? openRouterKey : groqKey;
}

init();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/config")
def config():
    """Tells the frontend whether server-side API keys are configured."""
    return jsonify({
        "openrouter_key_configured": bool(SERVER_OPENROUTER_API_KEY),
        "groq_key_configured": bool(SERVER_GROQ_API_KEY),
        "rate_limit_max": RATE_LIMIT_MAX,
        "rate_limit_window_hours": RATE_LIMIT_WINDOW // 3600,
    })


@app.route("/quota")
def quota():
    """Returns the current quota status for the requesting IP."""
    if not (SERVER_OPENROUTER_API_KEY or SERVER_GROQ_API_KEY):
        return jsonify({"limited": False, "remaining": None, "retry_after": 0})
    ip = _get_client_ip()
    now = time.time()
    with _rate_lock:
        timestamps = _rate_store.get(ip, [])
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        used = len(timestamps)
    remaining = max(0, RATE_LIMIT_MAX - used)
    retry_after = 0
    if remaining == 0 and timestamps:
        retry_after = int(RATE_LIMIT_WINDOW - (now - timestamps[0]))
    return jsonify({
        "limited": remaining == 0,
        "remaining": remaining,
        "max": RATE_LIMIT_MAX,
        "retry_after": retry_after,   # seconds until reset
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    provider = request.form.get("provider", "openrouter").strip()
    
    # Choose API key and URL depending on provider
    if provider == "groq":
        api_key = SERVER_GROQ_API_KEY or request.form.get("api_key", "").strip()
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        error_msg = "Please provide your Groq API key."
        has_server_key = bool(SERVER_GROQ_API_KEY)
    else:  # openrouter
        api_key = SERVER_OPENROUTER_API_KEY or request.form.get("api_key", "").strip()
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        error_msg = "Please provide your OpenRouter API key."
        has_server_key = bool(SERVER_OPENROUTER_API_KEY)

    if not api_key:
        return jsonify({"error": "Missing API key", "detail": error_msg}), 401

    # Rate limiting — only enforced when a server-side key is active for the chosen provider
    if has_server_key:
        ip = _get_client_ip()
        allowed, retry_after = _check_rate_limit(ip)
        if not allowed:
            hours = int(retry_after // 3600)
            minutes = int((retry_after % 3600) // 60)
            return jsonify({
                "error": "Rate limit reached",
                "detail": f"You have already reached your daily free analysis limit. Try again in {hours}h {minutes}m.",
                "retry_after": int(retry_after),
            }), 429

    model = request.form.get("model", "").strip()
    if not model:
        model = "openai/gpt-oss-20b"

    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "No file", "detail": "No PDF file was uploaded."}), 400

    filename = uploaded.filename or ""
    if not (uploaded.mimetype == "application/pdf" or filename.lower().endswith(".pdf")):
        return jsonify({"error": "Invalid file type", "detail": "Please upload a PDF file."}), 400

    pdf_bytes = uploaded.read()
    if len(pdf_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "File too large", "detail": "Max file size is 10 MB."}), 400

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    except Exception as exc:
        return jsonify({"error": "PDF error", "detail": f"Could not read the PDF: {exc}"}), 422

    if len(full_text.strip()) < 50:
        return (
            jsonify(
                {
                    "error": "Unreadable PDF",
                    "detail": (
                        "This PDF appears to be scanned or image-based. "
                        "Please use a text-based PDF."
                    ),
                }
            ),
            422,
        )

    extracted_text = full_text[:6000]

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document source: PDF upload\n\n{extracted_text}"},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = request.host_url.rstrip("/")
        headers["X-Title"] = "Mediq"

    try:
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                api_url,
                json=payload,
                headers=headers,
            )
    except httpx.RequestError as exc:
        return jsonify({"error": "Network error", "detail": str(exc)}), 502

    if resp.status_code == 401:
        return (
            jsonify(
                {
                    "error": "Invalid API key",
                    "detail": "Please check your OpenRouter key and try again.",
                }
            ),
            401,
        )
    if resp.status_code == 429:
        return (
            jsonify(
                {
                    "error": "Rate limit",
                    "detail": "Rate limit reached. Please wait a moment and try again.",
                }
            ),
            429,
        )
    if resp.status_code in (500, 529):
        return (
            jsonify(
                {
                    "error": "Service unavailable",
                    "detail": "The AI service is temporarily unavailable. Please try again.",
                }
            ),
            502,
        )
    if not resp.is_success:
        return (
            jsonify(
                {
                    "error": "API error",
                    "detail": f"OpenRouter returned status {resp.status_code}.",
                }
            ),
            502,
        )

    try:
        content_msg = resp.json()["choices"][0]["message"]["content"].strip()
        clean_content = _extract_json_block(content_msg)
        result = json.loads(clean_content, strict=False)
    except Exception as exc:
        print("Failed to parse LLM response:", resp.text)
        return (
            jsonify(
                {
                    "error": "Parse error",
                    "detail": "The AI returned an unexpected response. Please try again.",
                }
            ),
            502,
        )

    if has_server_key:
        ip = _get_client_ip()
        _record_rate_limit(ip)

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Mediq running → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
