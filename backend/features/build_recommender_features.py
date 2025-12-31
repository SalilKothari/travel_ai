import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# load data from csv's

users = pd.read_csv("../data/users_sim.csv")
destinations = pd.read_csv("../data/destinations_clean.csv")
interactions = pd.read_csv("../data/interactions_sim.csv")


# mapping action to labels:
action_map = {"view": 0, "click": 1, "book": 2}
interactions["label"] = interactions["action"].map(action_map)


#mapping for budget - ordinal data
budget_map = {"Budget": 1, "Mid-range": 2, "Luxury": 3}
users["budget_enc"] = users["budget_level"].map(budget_map)
destinations["budget_enc"] = destinations["budget_level"].map(budget_map)

# label encode countries:

# user home country
user_country_encoder = LabelEncoder()
users["home_country_enc"] = user_country_encoder.fit_transform(users["home_country"])

# dest country
dest_country_encoder = LabelEncoder()
destinations["country_enc"] = dest_country_encoder.fit_transform(destinations["country"])




# numeric features
USER_FEATURES = [
    "age", "budget_enc",
    "attr_culture", "attr_adventure", "attr_nature", "attr_beaches",
    "attr_nightlife", "attr_cuisine", "attr_wellness",
    "ctx_urban", "ctx_seclusion",
    "home_country_enc"
]

DEST_FEATURES = [
    "budget_enc",
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness",
    "urban", "seclusion",
    "country_enc"
]



df = interactions.merge(users[["user_id"] + USER_FEATURES], on="user_id", how="left")
df = df.merge(destinations[["dest_id"] + DEST_FEATURES], on="dest_id", how="left")


df.to_csv("recommender_features.csv", index=False)

print("Recommender features saved to recommender_features.csv")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())