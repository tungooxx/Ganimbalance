import numpy as np
import pandas as pd
# Importing some machine learning modules
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    print(f"Data shape: {data.shape}")
    print(data.Class.value_counts())

    # Drop missing values
    data.dropna(inplace=True)

    # Drop unnecessary columns
    data = data.drop(axis=1, columns="Time")

    # Scale Amount column
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])

    return data


def separate_classes(data):
    # Separate fraud and genuine transactions
    data_fraud = data[data.Class == 1]
    data_genuine = data[data.Class == 0]

    return data_fraud, data_genuine


def prepare_for_pca(data):
    # Prepare data for PCA
    X = data.drop(columns="Class", axis=1)
    y = data.Class

    return X, y


def apply_pca(X, y, n_components=2):
    # Apply PCA
    pca = PCA(n_components)
    transformed_data = pca.fit_transform(X)

    # Convert to DataFrame and add Class column
    ds = pd.DataFrame(transformed_data)
    ds['Class'] = y

    return ds


data = load_and_preprocess_data("Creditcard_dataset.csv")
data_fraud, data_genuine = separate_classes(data)
X, y = prepare_for_pca(data)
ds = apply_pca(X, y)

print(ds.head())

