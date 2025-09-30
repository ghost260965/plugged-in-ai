# app.py — PluggedIn AI (Showcase Build)
# Senior-level Streamlit dashboard with branding, caching, demo mode, and PDF export

import os
import io
import hashlib
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

# Local modules
from predictor import train_model, predict_next_days, load_model
from agent import generate_marketing_plan, chat_with_ai
from shopify_integration import fetch_products

# -----------------------------
# Page setup & global styling
# -----------------------------
st.set_page_config(page_title="PluggedIn AI — C4 Marketing", layout="wide")

CUSTOM_CSS = """
<style>
/* Brand palette */
:root {
  --brand:#e63946;
  --ink:#0b0f14;
  --muted:#6b7280;
  --panel:#0f172a14;
}

html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
h1,h2,h3 { letter-spacing:-0.02em; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.card {
  border: 1px solid #0f172a22; border-radius: 14px; padding: 16px 18px; background: white;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.card h3 { margin-top: 0.2rem; }
.small { color: var(--muted); font-size: 0.92rem; }
.kpi { font-weight: 700; font-size: 1.1rem; }
.footer { color:#94a3b8; text-align:center; margin-top: 14px; }
hr { border: none; border-top: 1px solid #0f172a1a; margin: 18px 0 12px; }
.brand { text-align:center; margin-bottom: 6px; }
.brand .logo { font-size: 34px; font-weight: 800; color: var(--brand); letter-spacing: -0.03em; }
.brand .tag { color: var(--muted); margin-top: -6px; }
.stButton>button { border-radius: 10px; font-weight: 600; }
.stDownloadButton>button { border-radius: 10px; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    "<div class='brand'><div class='logo'>⚡ PluggedIn AI</div>"
    "<div class='tag'>Predict your next best-sellers. Launch AI-powered campaigns. Grow.</div></div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers: caching & demo data
# -----------------------------
@st.cache_data(show_spinner=False)
def _hash_df(df: pd.DataFrame) -> str:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return hashlib.sha256(b.getvalue()).hexdigest()

@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    # 90-day demo with weekly seasonality + promotions
    dates = pd.date_range(end=datetime.today().date(), periods=90, freq="D")
    base = 60 + (dates.dayofweek * 2.5)
    promo = (pd.Series(range(len(dates))) % 13 == 0).astype(int) * 25
    noise = pd.Series(pd.Series(base).rolling(3, min_periods=1).mean()).shift(1).fillna(method="bfill")
    sales = (base + promo + (noise * 0.2)).round().astype(int)
    df = pd.DataFrame({"date": dates, "sales": sales, "is_promo": (promo > 0)})
    return df

@st.cache_data(show_spinner=False)
def train_cached(csv_bytes: bytes):
    # Train once for a given dataset
    tmp = io.BytesIO(csv_bytes)
    info = train_model(tmp)
    return info

@st.cache_data(show_spinner=False)
def forecast_cached(df: pd.DataFrame, n_days: int):
    return predict_next_days(df.copy(), n_days=n_days)

def require_openai_env() -> bool:
    has = bool(os.getenv("OPENAI_API_KEY"))
    if not has:
        st.warning("OPENAI_API_KEY not set. Add it in Streamlit Secrets for the live app.", icon="⚠️")
    return has

# -----------------------------
# Sidebar (inputs & data)
# -----------------------------
with st.sidebar:
    st.subheader("📂 Data")
    demo_mode = st.toggle("Use built-in demo data", value=True, help="Loads instantly with 90 days of sample sales.")
    uploaded = st.file_uploader("Or upload sales CSV (date,sales[,is_promo])", type=["csv"], accept_multiple_files=False)

    st.subheader("⚙️ Forecast Settings")
    horizon = st.slider("Days ahead", 7, 30, 14)

    st.subheader("🛒 Shopify (optional)")
    if st.button("Fetch Shopify products"):
        try:
            prods = fetch_products()
            if prods:
                st.success("Fetched Shopify products.")
                st.json(prods)
            else:
                st.info("No products or Shopify not configured.")
        except Exception as e:
            st.error(f"Shopify error: {e}")

# Choose data source
if demo_mode or uploaded is None:
    df_sales = load_demo_data()
else:
    try:
        df_sales = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# -----------------------------
# Layout: 3 panels
# -----------------------------
left, right = st.columns([2, 1])

# ========= Left: Forecast & Report =========
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 AI Demand Forecast")
    st.caption("Upload your data or use demo data. Train once → predict fast with caching.")

    st.dataframe(df_sales.head(), use_container_width=True)

    # Train model (cached by dataset bytes)
    try:
        dataset_key = _hash_df(df_sales)
        if st.button("🔨 Train Model", help="Trains a lightweight RandomForest on the current dataset"):
            with st.spinner("Training model…"):
                info = train_cached(df_sales.to_csv(index=False).encode("utf-8"))
                st.success(f"Model trained. MAE ≈ {info['mae']:.2f}")
        model = load_model()
    except Exception as e:
        st.error(f"Training error: {e}")
        model = None

    # Forecast
    fut_df = None
    if model:
        if st.button("🔮 Predict"):
            with st.spinner("Predicting future demand…"):
                try:
                    fut = forecast_cached(df_sales, horizon)
                    fut_df = pd.DataFrame(fut)
                    st.dataframe(fut_df, use_container_width=True)
                    fig = px.line(fut_df, x="date", y="predicted_sales",
                                  title="Predicted Sales (Next {} Days)".format(horizon))
                    st.plotly_chart(fig, use_container_width=True)

                    # "Top 3 pushes" — for per-SKU, expect a 'sku' column. Fallback: top forecast days.
                    st.markdown("#### 🚀 Top Push Windows")
                    top3 = fut_df.sort_values("predicted_sales", ascending=False).head(3)
                    st.table(top3.reset_index(drop=True))

                except Exception as e:
                    st.error(f"Forecast error: {e}")

    # PDF Export (text-only to avoid extra image deps)
    if st.button("📥 Export Report as PDF", disabled=fut_df is None):
        if fut_df is None:
            st.warning("Run a prediction first.", icon="⚠️")
        else:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(0, 10, "PluggedIn AI — Forecast Report", ln=True, align="C")
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(4)
            pdf.cell(0, 8, "Next {} Days — Predicted Sales:".format(horizon), ln=True)
            pdf.ln(2)
            for _, row in fut_df.iterrows():
                pdf.cell(0, 7, f"{row['date']}: {row['predicted_sales']}", ln=True)
            pdf.ln(6)
            # Include quick summary
            peak = fut_df.loc[fut_df['predicted_sales'].idxmax()]
            pdf.multi_cell(0, 7, f"Summary: Peak demand expected on {peak['date']} "
                                  f"with ~{int(round(peak['predicted_sales']))} units.")
            # Stream to download
            pdf_bytes = pdf.output(dest="S").encode("latin1")
            st.download_button("Download PDF", data=pdf_bytes, file_name="PluggedIn_Forecast.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= Middle: Campaign Launcher =========
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Campaign Launcher")
    st.caption("Generate a 7-day campaign plan with ad copy, visuals, CTA, and KPIs.")

    colA, colB = st.columns(2)
    with colA:
        product_name = st.text_input("Product / Offer", "Blue Snowboard Jacket")
    with colB:
        audience = st.text_input("Audience", "18–30 snowboarders in Whistler")

    if st.button("🎯 Generate Campaign Plan"):
        if not require_openai_env():
            st.stop()
        with st.spinner("Creating plan with the marketing model…"):
            try:
                plan = generate_marketing_plan(product_name, audience)
                st.markdown("#### Plan")
                st.write(plan)
            except Exception as e:
                st.error(f"OpenAI error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ========= Right: Advisor / Chat =========
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💬 PluggedIn Advisor")
    st.caption("Ask business questions. Answers are tuned for small-business operators.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_area("Ask PluggedIn AI…", height=120, placeholder="e.g., How do I price this jacket at 60% margin?")
    if st.button("Send"):
        if not require_openai_env():
            st.stop()
        if query.strip():
            with st.spinner("Thinking…"):
                try:
                    reply = chat_with_ai(query)
                    st.session_state.chat_history.append(("You", query))
                    st.session_state.chat_history.append(("AI", reply))
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

    # show chat history (latest first)
    for role, msg in st.session_state.chat_history[::-1]:
        if role == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='footer'>PluggedIn AI © 2025 • Built by C4 Marketing</div>", unsafe_allow_html=True)
