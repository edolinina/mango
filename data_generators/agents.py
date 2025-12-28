import pandas as pd
import numpy as np

def gen_finance(n=20):
    return pd.DataFrame({
        "department": np.random.choice(["IT","HR","Ops"], n),
        "monthly_cost": np.random.randint(50000,150000,n),
        "baseline_cost": np.random.randint(60000,160000,n),
        "revenue_impact": np.random.rand(n)
    })

def gen_it(n=20):
    return pd.DataFrame({
        "cpu_util": np.random.rand(n),
        "memory_util": np.random.rand(n),
        "cost_usd": np.random.randint(1000,15000,n)
    })

def gen_hr(n=20):
    return pd.DataFrame({
        "performance_score": np.random.uniform(2.5,5.0,n),
        "utilization": np.random.rand(n),
        "attrition_risk": np.random.rand(n)
    })

from sklearn.linear_model import LinearRegression

X = df[["monthly_cost","baseline_cost","revenue_impact"]]
y = df["baseline_cost"] - df["monthly_cost"]

model = LinearRegression().fit(X, y)

df["saving_potential"] = model.predict(X)
