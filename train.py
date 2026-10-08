import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

# Load the dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

# Drop PageValues: it is calculated from completed purchases, so it leaks the target
df = df.drop(columns=["PageValues"])

# Drop technical columns that barely change the purchase rate
df = df.drop(columns=["Browser", "OperatingSystems", "Region"])

# Merge rare traffic sources (fewer than 100 sessions) into one "other" group, coded 0
counts = df["TrafficType"].value_counts()
rare = counts[counts < 100].index
df.loc[df["TrafficType"].isin(rare), "TrafficType"] = 0

# Log-transform the page counts and durations: they have extreme values,
# and log1p (log of 1 + x) shrinks them while keeping 0 as 0
page_cols = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
]
df[page_cols] = np.log1p(df[page_cols])

# Flag sessions with a bounce rate of exactly 0: a large group that buys more often
df["NoBounce"] = (df["BounceRates"] == 0).astype(int)

# Turn the True/False columns into 1/0
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# One-hot encode the text columns: each value becomes its own 0/1 column
df = pd.get_dummies(df, columns=["Month", "VisitorType", "TrafficType"], drop_first=True, dtype=int)

# Separate the inputs (X) from the target (y)
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

# Stratified random split: 60% train, 20% validation, 20% test
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42
)

# Standardize: learn each column's mean and spread from the training group only,
# then apply the same transformation to validation and test
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Model 1: scikit-learn logistic regression
# C=np.inf turns regularization off, so all three versions solve the same problem
sk_model = LogisticRegression(C=np.inf, max_iter=1000)
sk_model.fit(X_train_s, y_train)

# Predicted probability of buying for each validation session
sk_val_prob = sk_model.predict_proba(X_val_s)[:, 1]

print("Validation log loss:", log_loss(y_val, sk_val_prob))
print("Validation AUC:", roc_auc_score(y_val, sk_val_prob))

# Turn the data into PyTorch tensors (PyTorch's own kind of number table)
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

# Model 2: manual PyTorch loop
# Start with all weights and the bias at zero
w = torch.zeros(X_train_t.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.5
epochs = 5000

for epoch in range(epochs):
    # 1. Predict: weighted sum, then sigmoid to get probabilities
    z = X_train_t @ w + b
    p = torch.sigmoid(z)

    # 2. Measure: log loss
    loss = -(y_train_t * torch.log(p) + (1 - y_train_t) * torch.log(1 - p)).mean()

    # 3. Find the slope: PyTorch computes the gradient of the loss for w and b
    loss.backward()

    # 4. Step: move each weight a little downhill, then reset the gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 1000 == 0:
        print("epoch", epoch, "loss", loss.item())

# Predicted probabilities on the validation group
with torch.no_grad():
    manual_val_prob = torch.sigmoid(X_val_t @ w + b).numpy()

print("Manual validation log loss:", log_loss(y_val, manual_val_prob))

# Model 3: standard PyTorch workflow (nn.Module + torch.optim)
torch.manual_seed(42)


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # One linear layer: holds the weights and the bias
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        # Returns the weighted sum (before the sigmoid)
        return self.linear(x).squeeze(1)


nn_model = LogisticRegressionModel(X_train_t.shape[1])
loss_fn = torch.nn.BCEWithLogitsLoss()  # sigmoid + log loss in one
optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()            # reset the gradients
    z = nn_model(X_train_t)          # 1. predict
    loss = loss_fn(z, y_train_t)     # 2. measure
    loss.backward()                  # 3. find the slope
    optimizer.step()                 # 4. step downhill

# Predicted probabilities on the validation group
with torch.no_grad():
    nn_val_prob = torch.sigmoid(nn_model(X_val_t)).numpy()

print("nn.Module validation log loss:", log_loss(y_val, nn_val_prob))