import pandas as pd
import random

data = []

for i in range(300):
    quantity = random.randint(10, 500)
    usage = random.randint(1, 50)
    days_left = random.randint(1, 200)
    cost = random.randint(1, 20)

    # logic-based labels (IMPORTANT — this teaches AI real rules)
    expiry_risk = 1 if days_left < 30 or usage > 30 else 0
    shortage_risk = 1 if quantity < usage * 5 else 0

    data.append([quantity, usage, days_left, cost, expiry_risk, shortage_risk])

df = pd.DataFrame(data, columns=[
    "quantity", "usage", "days_left", "cost",
    "expiry_risk", "shortage_risk"
])

df.to_csv("data.csv", index=False)

print("Dataset generated with 300 rows")