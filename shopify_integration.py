import os, requests
from dotenv import load_dotenv
load_dotenv()

SHOP_DOMAIN = os.getenv("SHOPIFY_STORE")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")

def fetch_products():
    if not SHOP_DOMAIN or not ACCESS_TOKEN: return []
    url = f"https://{SHOP_DOMAIN}/admin/api/2024-10/products.json?limit=5"
    headers = {"X-Shopify-Access-Token": ACCESS_TOKEN}
    r = requests.get(url, headers=headers)
    return r.json()
