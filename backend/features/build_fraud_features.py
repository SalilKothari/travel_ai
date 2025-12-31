''' 
 build _fraud_features.py
prepares ML features for fraud detection with user interactions and fraud sim data
'''

import pandas as pd
import numpy as np


# --------------------
# loading data
# --------------------

interactions = pd.read_csv('../data/interactions_sim.csv')
fraud = pd.read_csv('../data/fraud_sim.csv')


# --------------------
# Aggregating user level interaction features
# --------------------

agg = interactions.groupby("user_id").agg(
    total_views=("action", lambda x: (x=="view").sum()),
    total_clicks=("action", lambda x: (x=="click").sum()),
    total_books=("action", lambda x: (x=="book").sum()),
    total_actions=("action", "count")
).reset_index()

agg['click_rate'] = agg['total_clicks'] / (agg['total_views'] + 1e-6)
agg["book_rate"] = agg["total_books"] / (agg["total_actions"] + 1e-6)



# --------------------
# Merge with fraud table
# --------------------

fraud_features = fraud.merge(agg, on='user_id', how = 'left')

fraud_features[["total_views", "total_clicks", "total_books", "total_actions", "click_rate", "book_rate"]] = fraud_features[["total_views", "total_clicks", "total_books", "total_actions", "click_rate", "book_rate"]].fillna(0)

# --------------------
# Save to files
# --------------------

fraud_features.to_csv("fraud_features.csv", index=False)

print("Fraud features saved to fraud_features.csv")
print(f"Shape: {fraud_features.shape}")
print("Columns:", fraud_features.columns.tolist())