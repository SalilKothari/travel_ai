import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score
import mlflow
from typing import Dict, List, Tuple
import pickle
import os
import json


# Config
# -------------------------
class Config:
    # Model hyperparameters
    EMBEDDING_DIM = 32
    USER_HIDDEN_DIM = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.3
    
    # Training
    EPOCHS = 10
    BATCH_SIZE = 64
    NUM_NEGATIVES = 8
    PATIENCE = 3  # Early stopping
    
    # Recommendation
    NUM_CANDIDATES = 50
    TOP_K = 10
    
    # Paths
    MODEL_PATH = "models/hybrid_recommender.pth"
    PREPROCESSOR_PATH = "models/preprocessors.pkl"

config = Config()



# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




# Load data
# -------------------------
print("Loading data...")
users = pd.read_csv("../data/users_sim.csv")
destinations = pd.read_csv("../data/destinations_clean.csv")
interactions = pd.read_csv("../data/interactions_sim.csv")

# Create binary label
interactions["label"] = (interactions["action"] == "book").astype(int)


interactions["timestamp"] = pd.to_datetime(interactions["time"])

print(f"Users: {len(users)}, Destinations: {len(destinations)}, Interactions: {len(interactions)}")




# Parse avg_temp_monthly JSON and compute average
# -------------------------
def parse_temp_json(temp_str):
    """Parse temperature JSON and return average temperature"""
    try:
        temp_dict = json.loads(temp_str)
        avg_temps = [month_data['avg'] for month_data in temp_dict.values()]
        return np.mean(avg_temps)
    except:
        return np.nan

destinations['avg_temp'] = destinations['avg_temp_monthly'].apply(parse_temp_json)



# Encode destinations: 
# -------------------------
dest_encoder = LabelEncoder()
destinations["dest_idx"] = dest_encoder.fit_transform(destinations["dest_id"])
interactions["dest_idx"] = dest_encoder.transform(interactions["dest_id"])

NUM_DEST = destinations["dest_idx"].nunique()
print(f"Number of destinations: {NUM_DEST}")



# Time-based split (NO LEAKAGE)
# -------------------------
split_time = interactions["timestamp"].quantile(0.8)
train_df = interactions[interactions["timestamp"] <= split_time].copy()
val_df = interactions[interactions["timestamp"] > split_time].copy()

print(f"Train interactions: {len(train_df)}, Val interactions: {len(val_df)}")
print(f"Train positive rate: {train_df['label'].mean():.3f}")
print(f"Val positive rate: {val_df['label'].mean():.3f}")



# Feature definitions (based on actual data)
# -------------------------
# Budget level ordering (Low < Mid-range < Luxury) - ordinal data
BUDGET_ORDER = ['Budget', 'Mid-range', 'Luxury']

USER_NUMERIC = ['age']  # Age is numeric in the data
USER_THEMES = [
    'attr_culture', 'attr_adventure', 'attr_nature', 'attr_beaches',
    'attr_nightlife', 'attr_cuisine', 'attr_wellness', 'ctx_urban', 'ctx_seclusion'
]
USER_ORDINAL = ['budget_level']  # Ordinal encoding is better for ordered categories




DEST_NUMERIC = ['avg_temp']  # Computed average temperature
DEST_THEMES = [
    'culture', 'adventure', 'nature', 'beaches',
    'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion'
]
DEST_ORDINAL = ['budget_level']  # Same ordinal encoding

print(f"\nUser theme columns (numeric): {USER_THEMES}")
print(f"Destination theme columns (numeric): {DEST_THEMES}")





# Preprocessing Pipeline
# -------------------------
class FeaturePreprocessor:
    """Handles all feature preprocessing with proper train/test split"""
    
    def __init__(self):
        self.user_scaler = StandardScaler()
        self.dest_scaler = StandardScaler()
        self.user_ordinal = OrdinalEncoder(
            categories=[BUDGET_ORDER],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.dest_ordinal = OrdinalEncoder(
            categories=[BUDGET_ORDER],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.user_feat_map = None
        self.dest_feat_map = None
        self.user_features = None
        self.dest_features = None
        
    def fit_transform_users(self, users_df: pd.DataFrame, train_user_ids: np.ndarray) -> pd.DataFrame:
        """Fit on train users only, transform all"""
        users_df = users_df.copy()
        
        # Verify all theme columns are numeric
        for col in USER_THEMES:
            if col not in users_df.columns:
                raise ValueError(f"Missing column: {col}")
            if not pd.api.types.is_numeric_dtype(users_df[col]):
                raise ValueError(f"Column {col} is not numeric: {users_df[col].dtype}")
        
        # Fit scaler on train users only - avoid leaking info between train and test 
        train_mask = users_df['user_id'].isin(train_user_ids)
        numeric_cols = USER_NUMERIC + USER_THEMES
        self.user_scaler.fit(users_df.loc[train_mask, numeric_cols])
        
        # Transform all users
        users_df[numeric_cols] = self.user_scaler.transform(users_df[numeric_cols])
        
        # Ordinal encode budget_level (fit on train only)
        self.user_ordinal.fit(users_df.loc[train_mask, USER_ORDINAL])
        user_ordinal_encoded = self.user_ordinal.transform(users_df[USER_ORDINAL])
        
        # Combine features
        user_feat = pd.concat([
            users_df[['user_id'] + numeric_cols].reset_index(drop=True),
            pd.DataFrame(user_ordinal_encoded, columns=['budget_level_encoded'])
        ], axis=1)
        
        self.user_features = [col for col in user_feat.columns if col != 'user_id']
        self.user_feat_map = user_feat.set_index('user_id')
        
        print(f"\nUser features ({len(self.user_features)}): {self.user_features}")
        
        return user_feat
    
    def transform_user(self, user_dict: Dict) -> np.ndarray:
        """Transform a single user dictionary to feature vector"""
        user_df = pd.DataFrame([user_dict])
        
        # Ensure all required columns exist
        for col in USER_NUMERIC + USER_THEMES + USER_ORDINAL:
            if col not in user_df.columns:
                raise ValueError(f"Missing required feature: {col}")
        
        # Scale numeric features
        numeric_cols = USER_NUMERIC + USER_THEMES
        user_df[numeric_cols] = self.user_scaler.transform(user_df[numeric_cols])
        
        # Ordinal encode budget_level
        user_ordinal_encoded = self.user_ordinal.transform(user_df[USER_ORDINAL])
        
        # Combine
        user_vec = np.concatenate([
            user_df[numeric_cols].values,
            user_ordinal_encoded
        ], axis=1)
        
        return user_vec
    
    def fit_transform_destinations(self, dest_df: pd.DataFrame, train_dest_ids: np.ndarray) -> pd.DataFrame:
        """Fit on train destinations only, transform all"""
        dest_df = dest_df.copy()
        
        # Verify all theme columns are numeric
        for col in DEST_THEMES:
            if col not in dest_df.columns:
                raise ValueError(f"Missing column: {col}")
            if not pd.api.types.is_numeric_dtype(dest_df[col]):
                raise ValueError(f"Column {col} is not numeric: {dest_df[col].dtype}")
        
        # Handle missing temperature values
        if dest_df['avg_temp'].isna().any():
            print(f"Warning: {dest_df['avg_temp'].isna().sum()} destinations have missing temperature data")
            dest_df['avg_temp'].fillna(dest_df['avg_temp'].mean(), inplace=True)
        
        # Fit scaler on train destinations only
        train_mask = dest_df['dest_idx'].isin(train_dest_ids)
        numeric_cols = DEST_NUMERIC + DEST_THEMES
        self.dest_scaler.fit(dest_df.loc[train_mask, numeric_cols])
        
        # Transform all destinations
        dest_df[numeric_cols] = self.dest_scaler.transform(dest_df[numeric_cols])
        
        # Ordinal encode budget_level (fit on train only)
        self.dest_ordinal.fit(dest_df.loc[train_mask, DEST_ORDINAL])
        dest_ordinal_encoded = self.dest_ordinal.transform(dest_df[DEST_ORDINAL])
        
        # Combine features
        dest_feat = pd.concat([
            dest_df[['dest_idx'] + numeric_cols].reset_index(drop=True),
            pd.DataFrame(dest_ordinal_encoded, columns=['budget_level_encoded'])
        ], axis=1)
        
        self.dest_features = [col for col in dest_feat.columns if col != 'dest_idx']
        self.dest_feat_map = dest_feat.set_index('dest_idx')
        
        print(f"Destination features ({len(self.dest_features)}): {self.dest_features}")
        
        return dest_feat
    
    def save(self, path: str):
        """Save preprocessor to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        with open(path, 'wb') as f:
            pickle.dump(self, f)# using pickle for binary
    
    @staticmethod
    def load(path: str):
        """Load preprocessor from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)

# Initialize and fit preprocessor
preprocessor = FeaturePreprocessor()

train_user_ids = train_df['user_id'].unique()
train_dest_ids = train_df['dest_idx'].unique()

user_feat = preprocessor.fit_transform_users(users, train_user_ids)
dest_feat = preprocessor.fit_transform_destinations(destinations, train_dest_ids)

print(f"\nFinal user feature dim: {len(preprocessor.user_features)}")
print(f"Final dest feature dim: {len(preprocessor.dest_features)}")






# Candidate Generation (cached)
# -------------------------
class CandidateGenerator:
    """Fast candidate generation using pre-computed scores"""
    
    def __init__(self, users_df: pd.DataFrame, dest_df: pd.DataFrame):
        # Store raw features for matching
        self.users_raw = users_df.copy()
        self.dest_raw = dest_df.copy()
        self.match_cache = {}
    
    def get_candidates(self, user_dict: Dict, top_k: int = 50) -> List[int]:
        """Get top-k candidate destinations for a user"""
        # Extract user theme preferences (using actual column names)
        user_themes = np.array([
            user_dict.get('attr_culture', 0),
            user_dict.get('attr_adventure', 0),
            user_dict.get('attr_nature', 0),
            user_dict.get('attr_beaches', 0),
            user_dict.get('attr_nightlife', 0),
            user_dict.get('attr_cuisine', 0),
            user_dict.get('attr_wellness', 0),
            user_dict.get('ctx_urban', 0),
            user_dict.get('ctx_seclusion', 0)
        ])
        
        # Get destination themes
        dest_themes = self.dest_raw[DEST_THEMES].values
        
        # Compute dot product similarity (normalized by theme magnitudes)
        scores = np.dot(dest_themes, user_themes).astype(np.float64)
        
        # Budget matching bonus (ordinal comparison)
        user_budget = user_dict.get('budget_level', 'Mid-range')
        user_budget_idx = BUDGET_ORDER.index(user_budget) if user_budget in BUDGET_ORDER else 1
        
        dest_budget_idx = self.dest_raw['budget_level'].apply(
            lambda x: BUDGET_ORDER.index(x) if x in BUDGET_ORDER else 1
        ).values

        # Penalize large budget mismatches
        budget_penalty = np.abs(user_budget_idx - dest_budget_idx) * 0.5
        scores -= budget_penalty
        
        # Get top-k indices
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        
        return self.dest_raw.iloc[top_indices]['dest_idx'].tolist()

candidate_gen = CandidateGenerator(users, destinations)


# Dataset with pre-sampled negatives
# -------------------------
class RecomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: FeaturePreprocessor, 
                 num_neg: int = 8, seed: int = 42):
        self.pos = df[df['label'] == 1].reset_index(drop=True)
        self.num_neg = num_neg
        self.all_dest = destinations['dest_idx'].values
        self.preprocessor = preprocessor
        
        # Pre-sample negatives for consistency
        np.random.seed(seed)
        self.negative_samples = self._presample_negatives()
    
    def _presample_negatives(self) -> List[List[int]]:
        """Pre-sample negatives once for consistency"""
        negatives = []
        for idx in range(len(self.pos)): # iterate over every positive reaction
            row = self.pos.iloc[idx]
            negs = np.random.choice( # sample destinations that are not the positive destination 
                self.all_dest[self.all_dest != row['dest_idx']],
                size=self.num_neg,
                replace=False
            )
            negatives.append(negs.tolist())
        return negatives
    
    def __len__(self):
        return len(self.pos)
    
    def __getitem__(self, idx):
        row = self.pos.iloc[idx]
        samples = [(row['user_id'], row['dest_idx'], 1)]
        
        for neg_dest in self.negative_samples[idx]:
            samples.append((row['user_id'], neg_dest, 0))
        
        return samples

def collate_fn(batch):
    flat = [x for sub in batch for x in sub]
    u, d, y = zip(*flat)
    
    uf = torch.tensor(
        preprocessor.user_feat_map.loc[list(u)].values,
        dtype=torch.float32
    )
    df = torch.tensor(
        preprocessor.dest_feat_map.loc[list(d)].values,
        dtype=torch.float32
    )
    di = torch.tensor(d, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)
    
    return uf, df, di, y

train_dataset = RecomDataset(train_df, preprocessor, num_neg=config.NUM_NEGATIVES, seed=42)
val_dataset = RecomDataset(val_df, preprocessor, num_neg=config.NUM_NEGATIVES, seed=123)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# Improved Model with Dropout
# -------------------------
class HybridRecommender(nn.Module):
    def __init__(self, num_dest: int, user_dim: int, dest_dim: int, 
                 emb_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dest_emb = nn.Embedding(num_dest, emb_dim)
        
        self.user_mlp = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )
        
        self.dest_mlp = nn.Sequential(
            nn.Linear(dest_dim, emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, uf, df, di):
        uvec = self.user_mlp(uf)
        dvec = self.dest_emb(di) + self.dest_mlp(df)
        return (uvec * dvec).sum(dim=1)

model = HybridRecommender(
    NUM_DEST,
    user_dim=len(preprocessor.user_features),
    dest_dim=len(preprocessor.dest_features),
    emb_dim=config.EMBEDDING_DIM,
    hidden_dim=config.USER_HIDDEN_DIM,
    dropout=config.DROPOUT
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                            weight_decay=config.WEIGHT_DECAY)
loss_fn = nn.BCEWithLogitsLoss()

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2) # 0.5 - halving learning rate, patience - consectutive values with no improvement

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------
# Training with validation and early stopping
# -------------------------
def evaluate(model, dataloader):
    """Evaluate model on a dataset"""
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    
    with torch.no_grad():
        for uf, df, di, y in dataloader:
            uf, df, di, y = uf.to(device), df.to(device), di.to(device), y.to(device)
            logits = model(uf, df, di)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.sigmoid(logits).cpu().numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    avg_loss = total_loss / len(dataloader)
    
    return auc, avg_loss

print("\n" + "="*50)
print("Starting training...")
print("="*50)

mlflow.start_run()
mlflow.log_params({
    "model": "HybridRecommender",
    "embedding_dim": config.EMBEDDING_DIM,
    "hidden_dim": config.USER_HIDDEN_DIM,
    "learning_rate": config.LEARNING_RATE,
    "dropout": config.DROPOUT,
    "batch_size": config.BATCH_SIZE,
    "num_negatives": config.NUM_NEGATIVES
})

best_auc = 0
patience_counter = 0

for epoch in range(config.EPOCHS):
    # Training
    model.train()
    train_loss = 0
    
    for uf, df, di, y in train_loader:
        uf, df, di, y = uf.to(device), df.to(device), di.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(uf, df, di)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    val_auc, val_loss = evaluate(model, val_loader)
    train_auc, _ = evaluate(model, train_loader)
    
    avg_train_loss = train_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{config.EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    
    # Log to MLflow
    mlflow.log_metrics({
        "train_loss": avg_train_loss,
        "train_auc": train_auc,
        "val_loss": val_loss,
        "val_auc": val_auc
    }, step=epoch)
    
    # Learning rate scheduling
    scheduler.step(val_auc)
    
    # Early stopping and model checkpointing
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
        
        # Save best model
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
        }, config.MODEL_PATH)
        
        print(f"  ✓ New best model saved (AUC: {val_auc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print("\n" + "="*50)
print(f"Training complete! Best Val AUC: {best_auc:.4f}")
print("="*50)

mlflow.log_metric("best_val_auc", best_auc)
mlflow.end_run()

# Load best model
checkpoint = torch.load(config.MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

# Save preprocessor
preprocessor.save(config.PREPROCESSOR_PATH)
print(f"\nModel and preprocessor saved to {os.path.dirname(config.MODEL_PATH)}/")

# -------------------------
# Recommendation Function (ID-AGNOSTIC)
# -------------------------
def recommend_for_user(user_dict: Dict, top_k: int = 10, 
                       num_candidates: int = 50) -> pd.DataFrame:
    """
    Generate recommendations for a user based on their features.
    
    Args:
        user_dict: Dictionary with user features matching schema
                  Required keys: age, budget_level, attr_culture, attr_adventure,
                                attr_nature, attr_beaches, attr_nightlife, 
                                attr_cuisine, attr_wellness, ctx_urban, ctx_seclusion
        top_k: Number of recommendations to return
        num_candidates: Number of candidates to re-rank
    
    Returns:
        DataFrame with dest_id, city, country, and score
    """
    model.eval()
    
    # Generate candidates
    candidate_dest_ids = candidate_gen.get_candidates(user_dict, num_candidates)
    
    # Transform user features
    user_vec = preprocessor.transform_user(user_dict)
    
    # Prepare batch
    uf = torch.tensor(user_vec, dtype=torch.float32).repeat(len(candidate_dest_ids), 1).to(device)
    df = torch.tensor(
        preprocessor.dest_feat_map.loc[candidate_dest_ids].values,
        dtype=torch.float32
    ).to(device)
    di = torch.tensor(candidate_dest_ids, dtype=torch.long).to(device)
    
    # Get scores
    with torch.no_grad():
        scores = torch.sigmoid(model(uf, df, di)).cpu().numpy()
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'dest_idx': candidate_dest_ids,
        'score': scores
    })
    
    # Map back to destination IDs
    result_df = result_df.merge(
        destinations[['dest_idx', 'dest_id', 'city', 'country']],
        on='dest_idx'
    )
    
    # Sort and return top-k
    result_df = result_df.nlargest(top_k, 'score')
    
    return result_df[['dest_id', 'city', 'country', 'score']]



# -------------------------
# Testing
# -------------------------
print("\n" + "="*50)
print("Testing recommendation system...")
print("="*50)

# Test user profiles (matching actual data format)
test_users = [
    {
        "age": 28,
        "budget_level": "Budget",
        "attr_culture": 2,
        "attr_adventure": 4,
        "attr_nature": 5,
        "attr_beaches": 4,
        "attr_nightlife": 3,
        "attr_cuisine": 3,
        "attr_wellness": 2,
        "ctx_urban": 2,
        "ctx_seclusion": 4
    },
    {
        "age": 45,
        "budget_level": "Luxury",
        "attr_culture": 5,
        "attr_adventure": 1,
        "attr_nature": 2,
        "attr_beaches": 2,
        "attr_nightlife": 4,
        "attr_cuisine": 5,
        "attr_wellness": 4,
        "ctx_urban": 5,
        "ctx_seclusion": 1
    },
    {
        "age": 32,
        "budget_level": "Mid-range",
        "attr_culture": 3,
        "attr_adventure": 3,
        "attr_nature": 4,
        "attr_beaches": 5,
        "attr_nightlife": 2,
        "attr_cuisine": 4,
        "attr_wellness": 3,
        "ctx_urban": 2,
        "ctx_seclusion": 3
    }
]

for i, test_user in enumerate(test_users, 1):
    print(f"\n{'='*50}")
    print(f"Test User {i} (Age: {test_user['age']}, Budget: {test_user['budget_level']})")
    print(f"Top preferences: culture={test_user['attr_culture']}, "
          f"beaches={test_user['attr_beaches']}, "
          f"cuisine={test_user['attr_cuisine']}, "
          f"adventure={test_user['attr_adventure']}")
    
    recommendations = recommend_for_user(test_user, top_k=5)
    print("\nTop 5 Recommendations:")
    print(recommendations.to_string(index=False))

print("\n" + "="*50)
print("✓ System ready for production!")
print("="*50)
print("\nUsage example:")
print("""
new_user = {
    "age": 35,
    "budget_level": "Mid-range",  # Budget, Mid-range, or Luxury
    "attr_culture": 4,
    "attr_adventure": 3,
    "attr_nature": 4,
    "attr_beaches": 5,
    "attr_nightlife": 2,
    "attr_cuisine": 4,
    "attr_wellness": 3,
    "ctx_urban": 2,
    "ctx_seclusion": 4
}

recommendations = recommend_for_user(new_user, top_k=10)
""")