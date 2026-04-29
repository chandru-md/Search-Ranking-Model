import lightgbm as lgb
from data.load_istella import load_istella
from data.grouping import get_group

# Load data (use small subset first)
X_train, y_train, qid_train = load_istella("data/raw/train.txt", limit=20000)
group_train = get_group(qid_train)

# Create dataset
train_data = lgb.Dataset(X_train, label=y_train, group=group_train)

# Model parameters
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20
}

# Train model
model = lgb.train(params, train_data, num_boost_round=100)

# Save model
model.save_model("models/ltr_model.txt")

print("Model trained and saved successfully")