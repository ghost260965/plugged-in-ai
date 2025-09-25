import streamlit as st, pandas as pd, plotly.express as px
from predictor import train_model, predict_next_days, load_model
from agent import generate_marketing_plan, chat_with_ai
from shopify_integration import fetch_products
from fpdf import FPDF

# --- Branding ---
st.set_page_config(page_title="PluggedIn AI", layout="wide")
st.markdown("<h1 style='text-align:center; color:#e63946;'>⚡ PluggedIn AI ⚡</h1>", unsafe_allow_html=True)
st.caption("by C4 Marketing — Predict. Launch. Grow.")

# --- Sidebar ---
st.sidebar.header("📂 Data Input")
csv = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])
if st.sidebar.button("Generate Sample CSV"):
    import sample_data_gen

if st.sidebar.button("Fetch Shopify Products"):
    prods = fetch_products()
    st.sidebar.write(prods if prods else "No products / API not set")

# --- Main Layout ---
col1, col2 = st.columns([2,1])

# ---- Forecast Section ----
with col1:
    st.subheader("📈 AI Demand Forecast")
    if csv:
        df = pd.read_csv(csv)
        st.write("Preview of Uploaded Data")
        st.dataframe(df.head())

        if st.button("🔨 Train Forecast Model"):
            info = train_model(csv)
            st.success(f"✅ Model trained. MAE={info['mae']:.2f}")

        model = load_model()
        if model:
            days = st.slider("Days ahead", 7, 30, 14)
            if st.button("🔮 Predict Sales"):
                fut = predict_next_days(df, n_days=days)
                fut_df = pd.DataFrame(fut)

                st.write("Forecast Results")
                st.dataframe(fut_df)

                # Chart
                st.plotly_chart(px.line(fut_df, x="date", y="predicted_sales", title="Predicted Sales"), use_container_width=True)

                # Auto pick top 3
                top3 = fut_df.sort_values("predicted_sales", ascending=False).head(3)
                st.markdown("### 🚀 Top 3 Products to Push")
                st.table(top3)

                # Export PDF
                if st.button("📥 Export Report as PDF"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="PluggedIn AI Report", ln=True, align="C")
                    pdf.ln(10)
                    for _, row in fut_df.iterrows():
                        pdf.cell(200, 8, txt=f"{row['date']}: {row['predicted_sales']}", ln=True)
                    pdf.output("report.pdf")
                    with open("report.pdf", "rb") as file:
                        st.download_button("Download Report", file, file_name="PluggedIn_Report.pdf")
    else:
        st.info("Upload a sales CSV to train and forecast.")

    st.markdown("---")

    # ---- Campaign Launcher ----
    st.subheader("🚀 Campaign Launcher")
    prod = st.text_input("Product", "Blue Snowboard Jacket")
    aud = st.text_input("Audience", "18–30 snowboarders")
    if st.button("🎯 Generate Campaign Plan"):
        st.write(generate_marketing_plan(prod, aud))

# ---- Advisor Section ----
with col2:
    st.subheader("💬 PluggedIn Advisor")
    if "chat" not in st.session_state: st.session_state.chat=[]
    msg = st.text_area("Ask PluggedIn AI:", "")
    if st.button("Send"):
        if msg.strip():
            reply = chat_with_ai(msg)
            st.session_state.chat.append(("You", msg))
            st.session_state.chat.append(("AI", reply))
    for role,text in st.session_state.chat[::-1]:
        st.markdown(f"**{role}:** {text}")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>PluggedIn AI © 2025 • Powered by C4 Marketing</p>", unsafe_allow_html=True)
