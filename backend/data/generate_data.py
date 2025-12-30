import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -----------------------
# Reproducibility
# -----------------------
SEED = 123
np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# Global variables
# -----------------------
NUM_USERS = 600
DEST_PER_USER = 25
FRAUD_RATE = 0.03

DEST_PATH = "destinations.csv"

# -----------------------
# Load destinations
# -----------------------
destinations = pd.read_csv(DEST_PATH)
destinations["dest_id"] = destinations["id"]

ATTRACTION_FEATURES = [
    "culture", "adventure", "nature",
    "beaches", "nightlife", "cuisine", "wellness"
]

CONTEXT_FEATURES = ["urban", "seclusion"]

# -----------------------
# Destination statistics (used for realistic simulation)
# -----------------------
budget_distr = destinations["budget_level"].value_counts(normalize=True)

attraction_means = destinations[ATTRACTION_FEATURES].mean()
attraction_stds = destinations[ATTRACTION_FEATURES].std()

context_means = destinations[CONTEXT_FEATURES].mean()
context_stds = destinations[CONTEXT_FEATURES].std()

# -----------------------
# Simulate users
# -----------------------
users = []

for user_id in range(1, NUM_USERS + 1):

    # --- attraction preferences ---
    attraction_pref = {}
    for feat in ATTRACTION_FEATURES:
        attraction_pref[feat] = int(
            np.clip(
                np.random.normal(
                    attraction_means[feat], attraction_stds[feat]
                ),
                1,
                5,
            )
        )

    # --- context preferences ---
    context_pref = {}
    for feat in CONTEXT_FEATURES:
        context_pref[feat] = int(
            np.clip(
                np.random.normal(
                    context_means[feat], context_stds[feat]
                ),
                1,
                5,
            )
        )

    # --- base user fields ---
    user_row = {
        "user_id": user_id,
        "age": np.random.randint(18, 70),
        "home_country": np.random.choice(destinations["country"].unique()),
        "budget_level": np.random.choice(
            budget_distr.index, p=budget_distr.values
        ),
    }

    # --- add attraction features as columns ---
    for feat, val in attraction_pref.items():
        user_row[f"attr_{feat}"] = val

    # --- add context features as columns ---
    for feat, val in context_pref.items():
        user_row[f"ctx_{feat}"] = val

    users.append(user_row)

users = pd.DataFrame(users)


# -----------------------
# Interaction scoring
# -----------------------
def match_score(user, dest):
    score = 0.0

    # Budget match
    if dest["budget_level"] == user["budget_level"]:
        score += 2.0

    # Attraction alignment
    for feat in ATTRACTION_FEATURES:
        diff = abs(user[f"attr_{feat}"] - dest[feat])
        if diff <= 1:
            score += 2.0
        elif diff == 2:
            score += 1

    # Context alignment
    for feat in CONTEXT_FEATURES:
        diff = abs(user[f"ctx_{feat}"] - dest[feat])
        if diff <= 1:
            score += 3
        elif diff <= 2:
            score += 1.5

    return score


# -----------------------
# Generate interactions
# -----------------------
interactions = []
now = datetime.utcnow()

SCORE_THRESHOLD_HIGH = 17
SCORE_THRESHOLD_MID = 11

for _, user in users.iterrows():
    sampled_dests = destinations.sample(DEST_PER_USER, random_state=SEED)

    for _, dest in sampled_dests.iterrows():
        score = match_score(user, dest)

        if score >= SCORE_THRESHOLD_HIGH:
            action = np.random.choice(
                ["view", "click", "book"], p=[0.1, 0.3, 0.6]
            )
        elif score >= SCORE_THRESHOLD_MID:
            action = np.random.choice(
                ["view", "click", "book"], p=[0.3, 0.5, 0.2]
            )
        else:
            action = np.random.choice(
                ["view", "click", "book"], p=[0.6, 0.3, 0.1]
            )

        interactions.append(
            {
                "user_id": user["user_id"],
                "dest_id": dest["dest_id"],
                "action": action,
                "time": now
                - timedelta(minutes=np.random.randint(1, 60 * 24 * 30)),
            }
        )

interactions = pd.DataFrame(interactions)

# -----------------------
# Fraud simulation (separate table)
# -----------------------
num_fraud_users = int(FRAUD_RATE * NUM_USERS)
fraud_users = set(
    np.random.choice(users["user_id"], size=num_fraud_users, replace=False)
)

fraud_rows = []

for user_id in users["user_id"]:
    is_fraud = user_id in fraud_users

    fraud_rows.append(
        {
            "user_id": user_id,
            "clicks_per_min": np.random.randint(1, 6)
            if not is_fraud
            else np.random.randint(25, 60),
            "avg_session_time_min": np.random.uniform(3, 20)
            if not is_fraud
            else np.random.uniform(0.1, 1),
            "bookings_last_hour": np.random.randint(0, 2)
            if not is_fraud
            else np.random.randint(5, 20),
            "ip_entropy": np.random.uniform(0.6, 1.0)
            if not is_fraud
            else np.random.uniform(0.0, 0.3),
            "is_fraud": int(is_fraud),
        }
    )

fraud_df = pd.DataFrame(fraud_rows)

# -----------------------
# Save outputs
# -----------------------
users.to_csv("users_sim.csv", index=False)
interactions.to_csv("interactions_sim.csv", index=False)
fraud_df.to_csv("fraud_sim.csv", index=False)
destinations.to_csv("destinations_clean.csv", index=False)

print("✅ Data generation complete")
print(f"Users: {len(users)}")
print(f"Destinations: {len(destinations)}")
print(f"Interactions: {len(interactions)}")
print(f"Fraud rate: {fraud_df['is_fraud'].mean():.3f}")
