import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import mlflow
# from datetime import datetime


#-------------------------
# set up device
#-------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#-------------------------
# load data
#-------------------------


users = pd.read_csv('../data/users_sim.csv')
destinations = pd.read_csv('../data/destinations_clean.csv')
interactions = pd.read_csv('../data/interactions_sim.csv')

interactions['label'] = (interactions['action'] == 'book').astype(int)
interactions['timestamp'] = pd.to_datetime(interactions['time'])

#encode other data
dest_encoder = LabelEncoder()
destinations["dest_idx"] = dest_encoder.fit_transform(destinations["destination_id"])
interactions["dest_idx"] = dest_encoder.transform(interactions["destination_id"])

NUM_DEST = interactions['dest_idx'].nunique()



#------------------
# Splitting data - on time. 80-20 split. Avoid leaking data
#------------------

split_time = interactions['timestamp'].quantile(0.8)

train_df = interactions[interactions['timestamp'] <= split_time]
val_df = interactions[interactions['timestamp'] > split_time]



#------------------
# Building User Features
#------------------


USER_NUMERIC = ['budget']
USER_THEMES = [
    "pref_culture", "pref_adventure", "pref_nature", "pref_beaches",
    "pref_nightlife", "pref_cuisine", "pref_wellness", "pref_urban"
    ]

USER_DURATION = ["ideal_duration"]

user_scaler = StandardScaler()
users[USER_NUMERIC + USER_THEMES] = user_scaler.fit_transform(users[USER_NUMERIC + USER_THEMES])

user_dur_ohe = OneHotEncoder(sparse=False)
user_dur_encoded = user_dur_ohe.fit_transform(users[USER_DURATION])


user_feat = pd.concat(
    [
        users[["user_id"] + USER_NUMERIC + USER_THEMES].reset_index(drop=True),
        pd.DataFrame(user_dur_encoded, columns=user_dur_ohe.get_feature_names_out())
    ],
    axis=1
)

USER_FEATURES = user_feat.columns.drop("user_id")
uesr_feat_map = user_feat.set_index("user_id")



#------------------
# Building Destination Features
#------------------

DEST_NUMERIC = ["avg_cost", "avg_temp", "min_temp", "max_temp"]
DEST_THEMES = [
    "culture", "adventure", "nature", "beaches",
    "nightlife", "cuisine", "wellness", "urban"
]

DEST_CAT = ["budget_level", "ideal_duration"]

dest_scaler = StandardScaler()

destinations[DEST_NUMERIC + DEST_THEMES] = dest_scaler.fit_transform(destinations[DEST_NUMERIC + DEST_THEMES])

dest_cat_ohe = OneHotEncoder()
dest_cat_encoded = dest_cat_ohe.fit_transform(destinations[DEST_CAT])

dest_feat = pd.concat(
    [
        destinations[["dest_idx"] + DEST_NUMERIC + DEST_THEMES].reset_index(drop=True),
        pd.DataFrame(dest_cat_encoded, columns=dest_cat_ohe.get_feature_names_out())
    ],
    axis=1
)

DEST_FEATURES = dest_feat.columns.drop("dest_idx")
dest_feat_map = dest_feat.set_index("dest_idx")


# ------------------------------------
# Match score (candidate generation + explanation)
# ------------------------------------

def match_score(user_row, dest_row):
    score = 0.0
    score += np.dot(
        user_row[USER_THEMES].values,
        dest_row[DEST_THEMES].values
    )
    score -= abs(user_row["budget"] - dest_row["avg_cost"])
    return score


#------------------
# Dataset and Dataloader
#------------------

class RecomDataset(Dataset):

    def __init__(self, data):
        # each batch has 3 elements
        self.user = torch.tensor(data['user_idx'].values, dtype = torch.long)
        self.dest = torch.tensor(data['dest_idx'].values, dtype = torch.long)
        self.label = torch.tensor(data['label'].values, dtype = torch.float)



    def __len__(self):
        return len(self.user)
    
    def __getitem__(self, idx):
        return self.user[idx], self.dest[idx], self.label[idx]


dataset = RecomDataset(interactions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


#-----------------------
# Neural recommender model
#-----------------------


class NeuralRecommender(nn.Module):
    def __init__(self, num_users, num_dest, emb_size=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.dest_emb = nn.Embedding(num_dest, emb_size)
        self.fc = nn.Sequential(
            nn.Linear(emb_size*2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # nn.Sigmoid()
        )

    def forward(self, u, d):
        u_emb = self.user_emb(u)
        d_emb = self.dest_emb(d)
        x = torch.cat([u_emb, d_emb], dim=1)
        return self.fc(x).squeeze()


model = NeuralRecommender(NUM_USERS, NUM_DEST).to(device)
# later tweak hyper parameters
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
# using binary cross entropy loss for classification
lossfn = nn.BCEWithLogitsLoss()


#------------------------
# starting training
#------------------------


mlflow.start_run()
mlflow.log_param("model_type", "NeuralRecommender")
mlflow.log_param("embedding_size", 32)

EPOCHS = 5
for epoch in range(EPOCHS):
    epoch_loss = 0
    for u,d,l in loader: 
        u,d,l = u.to(device), d.to(device), l.to(device) # unpack to  device
        optimizer.zero_grad() # resets model params
        pred = model(u,d) # pass inputs in, forward pass
        loss = lossfn(pred, l)
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
    print(f"epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss/len(loader):.4f}")



#------------------------
# Evaluation
#------------------------

# turn off gradient calculation

with torch.no_grad():
    u = torch.tensor(interactions["user_idx"].values, dtype=torch.long).to(device)
    d = torch.tensor(interactions["dest_idx"].values, dtype=torch.long).to(device)
    y_true = interactions["label"].values
    y_pred = model(u, d).cpu().numpy()
    auc = roc_auc_score(y_true, y_pred)
    print(f"ROC_AUC: {auc:.4f}")
    mlflow.log_metric("roc_auc", auc)

mlflow.end_run()


#------------------------
# Recommend for User
#------------------------

def recommend_for_user(user_id, top_k=5):
    user_idx = torch.tensor([user_encoder.transform([user_id])[0]],dtype=torch.long).to(device)
    user_idx = user_idx.repeat(NUM_DEST) # for batch inference
    dest_idx = torch.tensor(np.arange(NUM_DEST),dtype=torch.long).to(device)

    # inference
    with torch.no_grad():
        scores = model(user_idx, dest_idx).cpu().numpy()
        top_idx = scores.argsort()[::-1][:top_k]
        recs = destinations.iloc[top_idx].copy()
        recs['score'] = scores[top_idx]
        return recs[["destination_id", "score"]] # TODO: double check, make sure destination_id is correct