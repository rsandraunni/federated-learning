'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import config


# Load dataset
data = pd.read_csv(config.DATA_PATH)

# Separate features and target
X = data.drop(config.TARGET_COLUMN, axis=1)
y = data[config.TARGET_COLUMN]

# Split into train/test first
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)

# Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframe
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df[config.TARGET_COLUMN] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df[config.TARGET_COLUMN] = y_test.values

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Save test set
test_df.to_csv("dataset/test_set.csv", index=False)

# Split training into hospitals
hospital_data = train_df.sample(frac=1, random_state=config.RANDOM_STATE)

split_size = len(hospital_data) // config.NUM_HOSPITALS

for i in range(config.NUM_HOSPITALS):
    start = i * split_size
    end = (i + 1) * split_size if i < config.NUM_HOSPITALS - 1 else len(hospital_data)
    hospital_data.iloc[start:end].to_csv(
        f"dataset/hospital_{i+1}.csv",
        index=False
    )

print("Datasets created successfully!")
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import config

SCALER_PATH = "models/scaler.pkl"

# Load dataset
data = pd.read_csv(config.DATA_PATH)

# Separate features and target
X = data.drop(config.TARGET_COLUMN, axis=1)
y = data[config.TARGET_COLUMN]

# Split into train/test first
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)

# Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframe
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df[config.TARGET_COLUMN] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df[config.TARGET_COLUMN] = y_test.values

# Create folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Save test set
test_df.to_csv("dataset/test_set.csv", index=False)

# Split training into hospitals
hospital_data = train_df.sample(frac=1, random_state=config.RANDOM_STATE)

split_size = len(hospital_data) // config.NUM_HOSPITALS

for i in range(config.NUM_HOSPITALS):
    start = i * split_size
    end = (i + 1) * split_size if i < config.NUM_HOSPITALS - 1 else len(hospital_data)
    hospital_data.iloc[start:end].to_csv(
        f"dataset/hospital_{i+1}.csv",
        index=False
    )

print("Datasets created successfully!")
print(f"Scaler saved successfully at: {SCALER_PATH}")
