import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

print(X_train_s.mean(axis=0).round(2)[:5])
print(X_train_s.std(axis=0).round(2)[:5])