# app.py — PluggedIn AI (CEO Demo Mode)
# A polished Streamlit dashboard for forecasting, campaign plans, video teasers, and PDF export
import os, io, math, tempfile, hashlib
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import imageio
import requests

from predictor import train_models, predict_next_days, top_products_forecast
from agent import generate_marketing_plan, chat_with_ai

# -----------------------------
# Page setup & styling
# -----------------------------
st.set_page_config(page_title="PluggedIn AI — C4 Marketing", layout="wide")

CSS = """
<style>
:root { --brand:#e63946; --ink:#0b0f14; --muted:#6b7280; }
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.card { border:1px solid #0f172a22; border-radius:14px; padding:16px 18px; background:white; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.brand { text-align:center; margin-bottom:8px; }
.brand .logo { font-size:34px; font-weight:800; color:var(--brand); letter-spacing:-.03em; }
.brand .tag { color:var(--muted); margin-top:-6px; }
.footer { color:#94a3b8; text-align:center; margin-top: 16px; }
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='brand'><div class='logo'>⚡ PluggedIn AI</div><div class='tag'>Predict best-sellers. Launch AI campaigns. Grow.</div></div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Demo product data (mock SKUs + images)
# -----------------------------
PRODUCTS = {
    "JACKET-BLUE": {
        "title": "Blue Snow Jacket",
        "image_url": "https://picsum.photos/id/1011/400/400"
    },
    "GOGGLES-PRO": {
        "title": "Pro Ski Goggles",
        "image_url": "https://picsum.photos/id/1012/400/400"
    },
    "HELMET-LITE": {
        "title": "Lite Snow Helmet",
        "image_url": "https://picsum.photos/id/1013/400/400"
    },
}

@st.cache_data(show_spinner=False)
def load_demo_df() -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today().date(), periods=90, freq="D")
    rows = []
    for pid, bump in [("JACKET-BLUE", 0), ("GOGGLES-PRO", 12), ("HELMET-LITE", -8)]:
        for i, d in enumerate(dates):
            base = 40 + (d.dayofweek * 3) + bump
            promo = 20 if i % 13 == 0 else 0
            noise = 5 * math.sin(i/5)
            sales = max(0, int(base + promo + noise))
            rows.append({"date": d.date(), "product_id": pid, "sales": sales, "is_promo": promo>0})
    return pd.DataFrame(rows)

def generate_simple_video(image_url: str, lines: list[str], seconds: int = 6, size=(720,720)) -> bytes:
    # fetch image
    try:
        r = requests.get(image_url, timeout=10)
        base_img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        base_img = Image.new("RGB", size, color=(245,245,245))
    base_img = base_img.resize(size)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        font_body  = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font_title = ImageFont.load_default()
        font_body  = ImageFont.load_default()

    frames = []
    total_frames = seconds * 24
    for t in range(total_frames):
        frame = base_img.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle([(0, size[1]-200), (size[0], size[1])], fill=(0,0,0,150))
        y = size[1]-180
        draw.text((30,y), lines[0][:30], fill=(230,57,70), font=font_title)
        y += 60
        for ln in lines[1:]:
            draw.text((30,y), ln[:40], fill=(255,255,255), font=font_body)
            y += 40
        frames.append(frame)

    clip = ImageSequenceClip([f for f in frames], fps=24)
    clip.write_videofile("out.mp4", codec="libx264", audio=False, verbose=False, logger=None)
    with open("out.mp4", "rb") as f:
        return f.read()

def create_promo_gif(product_name, image_paths, output_path="promo.gif"):
    frames = [Image.open(img).resize((400, 400)) for img in image_paths]
    imageio.mimsave(output_path, frames, duration=1.5)  # 1.5s per frame
    return output_path

# -----------------------------
# Forecast Section
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📈 AI Demand Forecast")
df = load_demo_df()
st.dataframe(df.head(), use_container_width=True)

if st.button("🔮 Predict & rank products"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    top3 = top_products_forecast(tmp.name, n_days=14, top_k=3)
    st.session_state["top3"] = top3
    st.success("Computed Top 3 products!")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Campaign Launcher
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🚀 Campaign Launcher")

top3 = st.session_state.get("top3", [])
if top3:
    cols = st.columns(3)
    for i, item in enumerate(top3):
        with cols[i]:
            pid = str(item["product_id"])
            title = PRODUCTS[pid]["title"]
            img = PRODUCTS[pid]["image_url"]
            avg_pred = round(item["avg_predicted_sales"],1)
            st.image(img, use_column_width=True)
            st.markdown(f"**{title}**  \nSKU: `{pid}`  \nForecast: ~{avg_pred}/day")

            if st.button(f"🎯 Plan {pid}"):
                plan = generate_marketing_plan(title, "18–30 outdoor shoppers")
                st.write(plan)
                st.session_state[f"plan_{pid}"] = plan

            if st.button(f"🎬 Video {pid}"):
                lines = [title, "Limited Drop", "Tap to Shop"]
                mp4 = generate_simple_video(img, lines)
                st.download_button("Download MP4", mp4, file_name=f"{pid}_promo.mp4", mime="video/mp4")
else:
    st.info("Click Predict & rank products first.")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Advisor + Report
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 💬 PluggedIn Advisor & PDF Report")

q = st.text_input("Ask a business question")
if st.button("Send"):
    if q.strip():
        a = chat_with_ai(q)
        st.session_state.setdefault("chat", []).append(("You",q))
        st.session_state["chat"].append(("AI",a))

if "chat" in st.session_state:
    for role,text in st.session_state["chat"][::-1]:
        st.markdown(f"**{role}:** {text}")

if st.button("📥 Export PDF"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0,10,"PluggedIn AI — Report", ln=True, align="C")
    pdf.set_font("Arial", size=11)
    pdf.cell(0,8,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(6)
    if top3:
        pdf.cell(0,8,"Top Products:", ln=True)
        for item in top3:
            pid = str(item["product_id"])
            title = PRODUCTS[pid]["title"]
            avg = round(item["avg_predicted_sales"],1)
            pdf.cell(0,7, f"{title} — ~{avg}/day", ln=True)
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    st.download_button("Download PDF", pdf_bytes, "PluggedIn_Summary.pdf", "application/pdf")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>PluggedIn AI © 2025 • Built by C4 Marketing</div>", unsafe_allow_html=True)
