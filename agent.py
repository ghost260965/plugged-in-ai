# agent.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_marketing_plan(product_name, audience, tone="energetic"):
    prompt = f"""
    You are a marketing assistant. Create a concise launch plan for '{product_name}'
    aimed at '{audience}'. Include campaign schedule, ad copy, visuals, and KPIs.
    Tone: {tone}.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful marketing assistant."},
            {"role":"user","content":prompt}
        ],
        max_tokens=600,
        temperature=0.7
    )
    return resp.choices[0].message.content

def chat_with_ai(user_message, context="You are a helpful small-business assistant."):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":context},
            {"role":"user","content":user_message}
        ],
        max_tokens=500,
        temperature=0.8
    )
    return resp.choices[0].message.content
