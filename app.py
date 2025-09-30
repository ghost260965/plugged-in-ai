import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from openai import OpenAI
from dotenv import load_dotenv
import plotly.express as px
import imageio
from PIL import Image
from fpdf import FPDF

# Load secrets
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- DEMO PRODUCTS --------
demo_products = [
    {"id": "SKU1", "name": "Snowboard Jacket", "images": ["demo1.png", "demo2.png"]},
    {"id": "SKU2", "name": "Winter Boots", "images": ["demo3.png", "demo4.png"]},
    {"id": "SKU3", "name": "Thermal Gloves", "images": ["demo5.png", "demo6.png"]},
]

# -------- GIF GENERATOR --------
def create_promo_gif(product_name, image_paths, output_path="promo.gif"):
    frames = [Image.open(img).resize((400, 400)) for img in image_paths]
    imageio.mimsave(output_path, frames, duration=1.5)
    return output_path

# -------- AI FUNCTIONS --------
def generate_marketing_plan(product, audience):
    prompt = f"Create a short marketing campaign plan for {product}, targeting {audience}."
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def chat_with_ai(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content.strip()

# -------- STREAMLIT UI --------
st.set_page_config(page_title="PluggedIn AI", layout="wide")
st.title("📊 PluggedIn AI — CEO Demo Mode")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("AI Demand Forecast")
    # Demo forecast data
    future_dates = pd.date_range(datetime.today(), periods=15).to_pydatetime().tolist()
    forecast_sales = np.random.randint(50, 150, size=15)
    df = pd.DataFrame({"date": future_dates, "forecast": forecast_sales})
    fig = px.line(df, x="date", y="forecast", title="Predicted Sales")
    st.plotly_chart(fig)

    st.subheader("Top Products to Promote")
    for p in demo_products:
        st.write(f"**{p['name']}** — predicted to grow 📈")
        gif = create_promo_gif(p["name"], p["images"])
        st.image(gif, caption=f"{p['name']} Promo GIF")

with col2:
    st.header("Campaign Launcher")
    product = st.selectbox("Select a product", [p["name"] for p in demo_products])
    audience = st.text_input("Target Audience", "18–30 snowboarders")
    if st.button("Generate Plan"):
        plan = generate_marketing_plan(product, audience)
        st.success("Generated Campaign Plan:")
        st.write(plan)

    st.header("PluggedIn Advisor")
    q = st.text_input("Ask a business question")
    if st.button("Ask AI"):
        answer = chat_with_ai(q)
        st.info(answer)

    st.header("Export Report")
    if st.button("Download PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "PluggedIn AI Demo Report\n\nForecast + Campaign Results")
        path = "report.pdf"
        pdf.output(path)
        with open(path, "rb") as f:
            st.download_button("Download Report", f, file_name="report.pdf")
