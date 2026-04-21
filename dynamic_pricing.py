import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
 

# -------------------------------
# STEP 1: Generate Data
# -------------------------------
np.random.seed(42)

n = 200

own_price = np.random.uniform(40, 100, n)
competitor_price = np.random.uniform(40, 100, n)

demand = 150 - 2*own_price + 1.5*competitor_price + np.random.normal(0, 5, n)
demand = np.maximum(demand, 1)

data = pd.DataFrame({
    "own_price": own_price,
    "competitor_price": competitor_price,
    "demand": demand
})

print("\nSample Data:\n", data.head())

# -------------------------------
# STEP 2: Train Model
# -------------------------------
X = data[["own_price", "competitor_price"]]
y = data["demand"]

model = RandomForestRegressor()
model.fit(X, y)

print("\nModel trained successfully ✅")

# -------------------------------
# STEP 3: Profit Function
# -------------------------------
cost_price = 30

def profit(price, competitor_price):
    demand = model.predict([[price, competitor_price]])[0]
    return (price - cost_price) * demand

# -------------------------------
# STEP 4: Find Best Price
# -------------------------------
prices = np.linspace(40, 100, 100)
profits = []

comp_price = 70

for p in prices:
    profits.append(profit(p, comp_price))

best_price = prices[np.argmax(profits)]
best_profit = max(profits)

print("\nBest Price:", round(best_price, 2))
print("Best Profit:", round(best_profit, 2))

# -------------------------------
# STEP 5: Plot Graph
# -------------------------------
plt.plot(prices, profits)
plt.xlabel("Price")
plt.ylabel("Profit")
plt.title("Profit vs Price")
plt.show()
print("\nGame Theory Simulation:")
competitor_prices = [50, 60, 70, 80, 90]

for cp in competitor_prices:
    temp_profits = []
    for p in prices:
        temp_profits.append(profit(p, cp))
    best_p = prices[np.argmax(temp_profits)]
    best_pf = max(temp_profits)
    print(f"Competitor Price = {cp} -> Our Best Price = {best_p:.2f}, Profit = {best_pf:.2f}")