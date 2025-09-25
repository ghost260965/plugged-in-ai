import pandas as pd
import numpy as np
from datetime import datetime, timedelta

dates = [datetime.today().date() - timedelta(days=x) for x in range(180)][::-1]
data = []
for d in dates:
    base = 50 + (d.timetuple().tm_yday % 30) * 0.8
    noise = np.random.randn() * 10
    promo = np.random.choice([0,1], p=[0.9,0.1]) * np.random.randint(20,50)
    sales = max(0, int(base + noise + promo))
    data.append({"date": d, "sales": sales, "is_promo": promo>0})

df = pd.DataFrame(data)
df.to_csv("sample_sales.csv", index=False)
print("✅ Wrote sample_sales.csv")
