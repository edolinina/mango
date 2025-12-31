import random
import datetime as dt
import numpy as np
import pandas as pd
import argparse

# -------------------- Finance --------------------
def gen_finance(seed=None, months=12):
    rng = np.random.default_rng(seed)
    rows = []
    for dept in range(1, 4):
        for m in range(months):
            rows.append({
                "department": f"Dept_{dept}",
                "date": (dt.date.today().replace(day=1) - dt.timedelta(days=30*m)),
                "spend": int(rng.uniform(50_000, 200_000)),
                "vendor": rng.choice(["Cloud", "Infra", "SaaS"]),
                "over_market": rng.uniform(-0.1, 0.2) # negative → cheaper than market, positive → overpriced
            })
    return pd.DataFrame(rows)

# -------------------- Ops --------------------
def gen_ops(seed=None, days=30):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, 5):
        for d in range(days):
            rows.append({
                "process": f"P{p}", # operational workflow ID
                "date": dt.date.today() - dt.timedelta(days=d),
                "throughput": rng.integers(50, 500), # units processed per day
                "utilization": rng.uniform(0.3, 1.0), # % of capacity used
                "lead_time": rng.uniform(1, 10) # time to complete the process
            })
    return pd.DataFrame(rows)

# -------------------- HR --------------------
def gen_hr(seed=None, n=50):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        hours = rng.uniform(20, 60)
        rows.append({
            "employee": f"E{i}",
            "hours": hours,
            "utilization": hours / 40, # workload vs standard 40h week
            "performance": rng.uniform(2, 5), # performance score
            "attrition_risk": rng.uniform(0, 1) # likelihood of leaving the company
        })
    return pd.DataFrame(rows)

def gen_ops_rl_episodes(
    episodes=50,
    steps=20,
    seed=None
):
    rng = np.random.default_rng(seed)
    data = []

    for ep in range(episodes):
        utilization = rng.uniform(0.4, 0.9)
        for t in range(steps):
            action = rng.uniform(-0.15, 0.15)  # capacity reallocation
            next_util = np.clip(utilization + action, 0, 1)

            reward = (
                (next_util - utilization)      # improvement
                - max(0, next_util - 0.95) * 0.5  # penalty if utilization > 95%
            )

            data.append({
                "episode": ep,
                "step": t,
                "state_utilization": utilization,
                "action": action,
                "reward": reward,
                "next_utilization": next_util
            })

            utilization = next_util

    return pd.DataFrame(data)


# -------------------- Registry --------------------
GENERATORS = {
    "finance": gen_finance,
    "ops": gen_ops,
    "hr": gen_hr,
    "ops_rl": gen_ops_rl_episodes
}

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", "-g", required=True, choices=GENERATORS.keys())
    parser.add_argument("--out", "-o",required=True, help="CSV output file")
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    df = GENERATORS[args.gen](seed=args.seed)
    print(df.head())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()

