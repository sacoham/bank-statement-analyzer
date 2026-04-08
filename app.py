import streamlit as st
import anthropic
import pdfplumber
import pandas as pd
import io
import json
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="🏦",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    border-left: 4px solid #1a56db;
  }
  .risk-low    { color: #0e7c42; font-weight: 700; font-size: 1.3rem; }
  .risk-medium { color: #b45309; font-weight: 700; font-size: 1.3rem; }
  .risk-high   { color: #b91c1c; font-weight: 700; font-size: 1.3rem; }
  .section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #374151;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.4rem;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
  }
  .signal-item { padding: 0.3rem 0; border-bottom: 1px solid #f3f4f6; }
  .tag {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
  }
  .tag-warn { background: #fef3c7; color: #92400e; }
  .tag-danger { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your key never leaves your session.",
    )
    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Upload a bank statement (PDF or CSV) and get an AI-powered "
        "underwriting summary — key financials, risk signals, and a credit memo."
    )
    st.markdown("---")
    st.markdown("**Sample CSV format**")
    st.caption("Date, Description, Amount, Balance")
    st.caption("2024-01-03, Payroll deposit, 8500.00, 12340.50")
    st.caption("2024-01-05, Rent payment, -2200.00, 10140.50")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏦 Bank Statement Analyzer")
st.markdown(
    "Upload a business bank statement and receive an AI-generated underwriting "
    "assessment: key financial metrics, risk signals, and a credit memo summary."
)
st.markdown("---")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload bank statement",
    type=["pdf", "csv"],
    help="PDF or CSV format. CSV should have columns: Date, Description, Amount, Balance.",
)

# ── Extraction helpers ────────────────────────────────────────────────────────
def extract_pdf_text(file) -> str:
    text_blocks = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_blocks.append(text)
            table = page.extract_table()
            if table:
                for row in table:
                    if row:
                        text_blocks.append(" | ".join(str(c) for c in row if c))
    return "\n".join(text_blocks)


def extract_csv_text(file) -> str:
    df = pd.read_csv(file)
    return df.to_string(index=False)


# ── Claude analysis ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior underwriter at a business lender.
You analyze bank statements to assess creditworthiness for business loan applications.

Your output must be a single valid JSON object with exactly these keys:

{
  "business_name": "string — infer from payee names or use 'Business Account'",
  "statement_period": "string — e.g. 'January–March 2024'",
  "risk_rating": "Low | Medium | High",
  "risk_rationale": "1–2 sentences explaining the rating",
  "metrics": {
    "avg_monthly_inflow": "dollar amount as string",
    "avg_monthly_outflow": "dollar amount as string",
    "avg_ending_balance": "dollar amount as string",
    "lowest_balance": "dollar amount as string",
    "net_cash_flow_trend": "Positive | Neutral | Negative",
    "months_analyzed": "integer as string"
  },
  "risk_signals": [
    {
      "severity": "Low | Medium | High",
      "signal": "short label",
      "detail": "one sentence explanation"
    }
  ],
  "positive_indicators": ["list of positive observations as strings"],
  "credit_memo": "3–4 paragraph professional credit memo summarizing the account, key strengths, key concerns, and a recommendation. Write in the style of a senior credit analyst."
}

Be concise, accurate, and professional. If data is limited, make reasonable inferences and note them.
Return ONLY valid JSON — no markdown, no code fences, no extra text."""


def analyze_statement(raw_text: str, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Analyze this bank statement and return your JSON assessment:\n\n{raw_text[:12000]}",
            }
        ],
    )
    raw = message.content[0].text.strip()
    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── Main flow ─────────────────────────────────────────────────────────────────
if uploaded_file and api_key:
    with st.spinner("Extracting data from document…"):
        if uploaded_file.name.endswith(".pdf"):
            raw_text = extract_pdf_text(uploaded_file)
        else:
            raw_text = extract_csv_text(uploaded_file)

    if not raw_text.strip():
        st.error("Could not extract text from the file. Try a different format.")
        st.stop()

    with st.spinner("Running AI underwriting analysis…"):
        try:
            result = analyze_statement(raw_text, api_key)
        except json.JSONDecodeError as e:
            st.error(f"Could not parse AI response. Try again. ({e})")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── Results ──────────────────────────────────────────────────────────────
    col_header, col_rating = st.columns([3, 1])
    with col_header:
        st.markdown(f"### {result.get('business_name', 'Business Account')}")
        st.caption(f"Statement period: {result.get('statement_period', 'N/A')}  ·  Analyzed {datetime.now().strftime('%b %d, %Y')}")
    with col_rating:
        rating = result.get("risk_rating", "N/A")
        css_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(rating, "")
        st.markdown(f"<div style='text-align:right; padding-top:0.5rem;'>Risk Rating<br><span class='{css_class}'>{rating}</span></div>", unsafe_allow_html=True)

    st.caption(f"_{result.get('risk_rationale', '')}_")
    st.markdown("---")

    # ── Metrics ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Financial Metrics</div>", unsafe_allow_html=True)
    metrics = result.get("metrics", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Avg Monthly Inflow",  metrics.get("avg_monthly_inflow", "—"))
    with c2:
        st.metric("Avg Monthly Outflow", metrics.get("avg_monthly_outflow", "—"))
    with c3:
        st.metric("Avg Ending Balance",  metrics.get("avg_ending_balance", "—"))
    with c4:
        st.metric("Lowest Balance",      metrics.get("lowest_balance", "—"))
    with c5:
        trend = metrics.get("net_cash_flow_trend", "—")
        trend_emoji = {"Positive": "📈", "Neutral": "➡️", "Negative": "📉"}.get(trend, "")
        st.metric("Cash Flow Trend", f"{trend_emoji} {trend}")

    # ── Two-column layout for signals + positives ─────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'>⚠️ Risk Signals</div>", unsafe_allow_html=True)
        signals = result.get("risk_signals", [])
        if signals:
            for s in signals:
                sev = s.get("severity", "Low")
                tag_class = {"High": "tag-danger", "Medium": "tag-warn"}.get(sev, "tag")
                st.markdown(
                    f"<div class='signal-item'>"
                    f"<span class='{tag_class} tag'>{sev}</span>"
                    f"<strong>{s.get('signal', '')}</strong><br>"
                    f"<span style='color:#6b7280;font-size:0.88rem'>{s.get('detail', '')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No significant risk signals detected.")

    with col_right:
        st.markdown("<div class='section-header'>✅ Positive Indicators</div>", unsafe_allow_html=True)
        positives = result.get("positive_indicators", [])
        if positives:
            for p in positives:
                st.markdown(f"- {p}")
        else:
            st.info("No standout positives noted.")

    # ── Credit Memo ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📄 Credit Memo</div>", unsafe_allow_html=True)
    memo = result.get("credit_memo", "")
    for para in memo.split("\n"):
        if para.strip():
            st.markdown(para)

    # ── Raw text expander ─────────────────────────────────────────────────────
    with st.expander("View extracted statement text"):
        st.text(raw_text[:5000])

elif uploaded_file and not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to run the analysis.")

elif not uploaded_file:
    # Landing state
    st.markdown("#### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.markdown("Drop in a PDF or CSV bank statement — personal, business, or SMB.")
    with col2:
        st.markdown("**2. Analyze**")
        st.markdown("Claude reads the statement and extracts financial signals the way a senior underwriter would.")
    with col3:
        st.markdown("**3. Review**")
        st.markdown("Get a structured credit memo: metrics, risk signals, and a recommendation — in seconds.")
