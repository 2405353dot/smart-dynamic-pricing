import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Smart Dynamic Pricing", layout="wide")

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title {
        font-size: 42px;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        text-align: center;
    }
    .card-title {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .card-value {
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Smart Dynamic Pricing System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine Learning + Decision Systems + Competitive Pricing Intelligence</div>',
    unsafe_allow_html=True
)

# -----------------------------
# TRAINING DATA
# -----------------------------
np.random.seed(42)
n = 300

own_price = np.random.uniform(40, 100, n)
competitor_price = np.random.uniform(40, 100, n)
marketing_spend = np.random.uniform(5, 50, n)

demand = (
    160
    - 2.2 * own_price
    + 1.6 * competitor_price
    + 0.8 * marketing_spend
    + np.random.normal(0, 5, n)
)

demand = np.maximum(demand, 1)

data = pd.DataFrame({
    "own_price": own_price,
    "competitor_price": competitor_price,
    "marketing_spend": marketing_spend,
    "demand": demand
})

X = data[["own_price", "competitor_price", "marketing_spend"]]
y = data["demand"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

cost_price = 30

def predict_demand(price, comp_price, marketing):
    return model.predict([[price, comp_price, marketing]])[0]

def profit(price, comp_price, marketing):
    d = predict_demand(price, comp_price, marketing)
    return (price - cost_price) * d

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Input Parameters")
user_price = st.sidebar.slider("Your Price", 40, 100, 60)
user_comp_price = st.sidebar.slider("Competitor Price", 40, 100, 70)
user_marketing = st.sidebar.slider("Marketing Spend", 5, 50, 20)

predicted_demand = predict_demand(user_price, user_comp_price, user_marketing)
predicted_profit = profit(user_price, user_comp_price, user_marketing)

prices = np.linspace(40, 100, 120)
profits = [profit(p, user_comp_price, user_marketing) for p in prices]

best_price = prices[np.argmax(profits)]
best_profit = max(profits)

# -----------------------------
# KPI CARDS
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="card"><div class="card-title">Selected Price</div><div class="card-value">{user_price:.2f}</div></div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f'<div class="card"><div class="card-title">Predicted Demand</div><div class="card-value">{predicted_demand:.2f}</div></div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f'<div class="card"><div class="card-title">Predicted Profit</div><div class="card-value">{predicted_profit:.2f}</div></div>',
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f'<div class="card"><div class="card-title">Recommended Price</div><div class="card-value">{best_price:.2f}</div></div>',
        unsafe_allow_html=True
    )

st.write("")

# -----------------------------
# CHART + SUMMARY
# -----------------------------
left, right = st.columns([2, 1])

with left:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices, profits)
    ax.axvline(best_price, linestyle="--")
    ax.set_xlabel("Your Price")
    ax.set_ylabel("Expected Profit")
    ax.set_title("Profit vs Price")
    st.pyplot(fig)

with right:
    st.subheader("Decision Summary")
    st.write(f"**Your selected price:** {user_price:.2f}")
    st.write(f"**Competitor price:** {user_comp_price:.2f}")
    st.write(f"**Marketing spend:** {user_marketing:.2f}")
    st.write(f"**Expected demand:** {predicted_demand:.2f}")
    st.write(f"**Expected profit:** {predicted_profit:.2f}")
    st.write(f"**Best suggested price:** {best_price:.2f}")
    st.write(f"**Maximum expected profit:** {best_profit:.2f}")

# -----------------------------
# COMPETITOR ANALYSIS
# -----------------------------
st.markdown("---")
st.subheader("Competitor Reaction Analysis")

competitor_prices = [50, 60, 70, 80, 90]
results = []

for cp in competitor_prices:
    temp_profits = [profit(p, cp, user_marketing) for p in prices]
    temp_best_price = prices[np.argmax(temp_profits)]
    temp_best_profit = max(temp_profits)

    results.append({
        "Competitor Price": cp,
        "Recommended Our Price": round(temp_best_price, 2),
        "Expected Profit": round(temp_best_profit, 2)
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

# -----------------------------
# DOWNLOAD RESULTS
# -----------------------------
st.markdown("---")
csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Competitor Analysis CSV",
    data=csv,
    file_name="competitor_analysis.csv",
    mime="text/csv"
)

with st.expander("Show Sample Training Data"):
    st.dataframe(data.head(20), use_container_width=True)

st.caption("Internship portfolio project: Machine Learning + Decision Systems + Pricing Strategy")