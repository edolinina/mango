import datetime as dt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

DEPARTMENTS = ["Core", "SaaS", "AI Products"]
SERVICES = ["Auth", "Cloud", "AIEngine"]
MONTHS = 12
EMPLOYEES_NUM = 100
DAYS = 15

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_SOURCES_DIR = BASE_DIR / "data_sources"


# -------------------- Business --------------------
def gen_business(seed=None, months=MONTHS):
    rng = np.random.default_rng(seed)
    rows = []
    for dept in DEPARTMENTS:
        for m in range(1, months + 1):
            revenue = rng.uniform(200_000, 800_000)
            cost = revenue * rng.uniform(0.4, 0.7)
            hr_spend = revenue * rng.uniform(0.15, 0.30)
            operational_cost = revenue * rng.uniform(0.15, 0.35)
            product_margin = revenue - hr_spend - operational_cost
            active_customers = int(revenue / rng.uniform(50, 150))
            rows.append({
                "department": dept,
                "month": m,
                "revenue": round(revenue, 2),
                "hr_spend": round(hr_spend, 2),
                "operational_cost": round(operational_cost, 2),
                "product_margin": round(product_margin, 2),
                "active_customers": active_customers,
                "delivery_risk": rng.integers(0, 3), # 0: low, 1: medium, 2: high
            })
    return pd.DataFrame(rows)


# -------------------- IT --------------------
def gen_it(seed=None, days: int = DAYS):
    rng = np.random.default_rng(seed)
    rows = []
    for svc in SERVICES:
        unit_cost = rng.uniform(0.03, 0.06) # base cost per capacity unit
        for d in range(1, days + 1):
            traffic = int(rng.integers(1_000, 50_000))
            capacity = int(rng.integers(2_000, 60_000))
            cost = capacity * float(rng.uniform(0.02, 0.06))
            infra_cost = capacity * unit_cost * (1 + max(0, capacity \
                - traffic)/traffic) # penalty for over-provisioning

            rows.append({
                "service": svc,
                "day": d,
                "traffic": traffic,
                "capacity": capacity,
                "infra_cost": round(infra_cost, 2),
                "sla_met": int(capacity >= traffic)
            })
    return pd.DataFrame(rows)


# -------------------- HR --------------------
def gen_hr(seed=None, employee_num=EMPLOYEES_NUM):
    rng = np.random.default_rng(seed)
    rows = []

    # distribute employees evenly across departments
    per_dept = employee_num // len(DEPARTMENTS)
    remainder = employee_num % len(DEPARTMENTS)

    # role distribution
    role_levels = [1, 2, 3, 4]  # junior → lead
    role_probs = [0.35, 0.35, 0.2, 0.1]  # fewer leads

    emp_id = 0
    for dept_i, dept in enumerate(DEPARTMENTS):
        n = per_dept + (1 if dept_i < remainder else 0)

        # delivery impact depends on role
        delivery_impact = {
            1: rng.integers(1, 3),
            2: rng.integers(2, 5),
            3: rng.integers(4, 7),
            4: rng.integers(6, 10),
        }

        for _ in range(n):
            role = rng.choice(role_levels, p=role_probs)
            rows.append({
                "employee": f"E{emp_id}",
                "department": dept,
                "role_level": role,
                "delivery_impact": delivery_impact[role],
                "performance_grade": rng.integers(2, 5),
                "contribution_rate": rng.integers(3, 10),
            })
            emp_id += 1

    return pd.DataFrame(rows)


# -------------------- IT RL Episodes --------------------
def gen_it_rl_episodes(episodes=50, steps=30, seed=None):
    rng = np.random.default_rng(seed)
    rows = []

    for ep in range(episodes):
        unit_cost = rng.uniform(0.03, 0.06) # base cost per capacity unit
        capacity = rng.integers(10_000, 60_000)

        traffic = int(rng.integers(10_000, 50_000))
        for t in range(steps):
            traffic = int(np.clip(
                traffic + rng.integers(-3000, 3000),
                8_000, 50_000
            ))

            # actions = scale down / keep / scale up
            action = rng.choice([-5_000, 0, 5_000])
            next_capacity = int(np.clip(capacity + action, 5_000, 70_000))

            sla_met = int(next_capacity >= traffic)

            next_cost = next_capacity * unit_cost
            prev_cost = capacity * unit_cost

            # reward: cost reduction + SLA constraint
            reward = (prev_cost - next_cost)
            reward += 50 if sla_met else -100

            rows.append({
                "episode": ep,
                "step": t,
                "traffic": traffic,
                "capacity": capacity,
                "action": action,
                "next_capacity": next_capacity,
                "infra_cost": round(next_cost, 2),
                "sla_met": sla_met,
                "reward": round(reward, 2),
            })

            capacity = next_capacity

    return pd.DataFrame(rows)


# -------------------- Registry --------------------
GENERATORS = {
    "it": gen_it,
    "hr": gen_hr,
    "business": gen_business,
    "it_rl": gen_it_rl_episodes
}

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", "-g", required=True, choices=GENERATORS.keys())
    parser.add_argument("--out", required=False, help="CSV output file")
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    df = GENERATORS[args.gen](seed=args.seed)
    print(df.head(10))

    out_path = None
    if not args.out:
        out_path = DATA_SOURCES_DIR / f"{args.gen}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

