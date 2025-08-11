# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 21:58:23 2025

@author: PC
"""
from tensorflow.keras import layers
import tensorflow as tf
from pathlib import Path
import pandas as pd
from helperFunctions import *


def load_financial_data():

    output_path = "data.csv"
    # Load transactions in chunks
    transactions_chunks = pd.read_csv(
        Path("../datasets/FinancialData/transactions_data.csv"),
        chunksize=500_000
    )

    # Load small mcc dataset
    mcc_codes_series = pd.read_json(
        Path("../datasets/FinancialData/mcc_codes.json"),
        typ='series'
    )
    mcc_codes = mcc_codes_series.reset_index()
    mcc_codes.columns = ['mcc', 'description']

    train_fraud_labels = pd.read_json(
        Path("../datasets/FinancialData/train_fraud_labels.json")
    )
    train_fraud_labels = train_fraud_labels.reset_index()
    train_fraud_labels.columns = ['id', 'label']
    train_fraud_labels['id'] = train_fraud_labels['id'].astype('int64')

    # Load all card, users data once
    cards_data = pd.read_csv(Path("../datasets/FinancialData/cards_data.csv"))
    cards_data = cards_data.rename(columns={'id': 'card_id'})
    cards_data['card_id'] = cards_data['card_id'].astype('int64')
    users_data = pd.read_csv(Path("../datasets/FinancialData/users_data.csv"))
    users_data = users_data.rename(columns={'id': 'client_id'})

    latitudes = users_data['latitude']
    longitudes = users_data['longitude']

    cluster_labels = calcule_kmeans(latitudes, longitudes)

    users_data['cluster_label'] = cluster_labels
    first_chunk = True

    for chunk in transactions_chunks:
        chunk = pd.merge(chunk, mcc_codes, on='mcc', how='left')
        chunk = pd.merge(chunk, train_fraud_labels, on='id', how='left')
        chunk = pd.merge(chunk, cards_data, on=[
            'client_id', 'card_id'], how='left')
        chunk = pd.merge(chunk, users_data, on='client_id', how='left')
        mode = 'w' if first_chunk else 'a'
        chunk.to_csv("data.csv", mode=mode, header=first_chunk, index=False)

        first_chunk = False  # Set the flag to False after the first write.

    print(f"All data successfully merged and saved to {output_path}.")
    return None


# load_financial_data()
csv_file = "data.csv"

dataset = tf.data.experimental.make_csv_dataset(
    file_pattern=csv_file,
    batch_size=32,  # choose your batch size
    label_name='label',
    num_epochs=1,
    shuffle=True,
    shuffle_seed=42
)


dataset_size = 415809

# Define your desired split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Calculate the number of elements for each split
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

train_batches = train_size // 32
val_batches = val_size // 32

# Create the training dataset
train_dataset = dataset.take(train_batches)


# Create the validation dataset by skipping the training set and taking the next part
val_dataset = dataset.skip(train_batches).take(val_batches)

# Create the test dataset by skipping both the training and validation sets
test_dataset = dataset.skip(train_batches + val_batches)


def dollar_to_float(X):
    value = tf.strings.regex_replace(X, r'[$]', '')
    return tf.strings.to_number(value, tf.float32)


def transaction_date_to_int(X):
    year = tf.strings.substr(X, 0, 4)
    month = tf.strings.substr(X, 5, 2)
    day = tf.strings.substr(X, 8, 2)
    hour = tf.strings.substr(X, 11, 2)
    minute = tf.strings.substr(X, 14, 2)
    second = tf.strings.substr(X, 17, 2)

    # Convert to numbers
    return {
        'year': tf.strings.to_number(year, tf.int32),
        'month': tf.strings.to_number(month, tf.int32),
        'day': tf.strings.to_number(day, tf.int32),
        'hour': tf.strings.to_number(hour, tf.int32),
        'minute': tf.strings.to_number(minute, tf.int32),
        'second': tf.strings.to_number(second, tf.int32),
    }


def month_year_date_to_int(X):
    month = tf.strings.substr(X, 0, 2)
    year = tf.strings.substr(X, 3, 2)
    # Convert to numbers
    return {
        'month': tf.strings.to_number(month, tf.int32),
        'year': tf.strings.to_number(year, tf.int32),
    }


# def transaction_new_features_names(function_transformer, feature_names_in):
#     return [
#         'year',
#         'month',
#         'day',
#         'hour',
#         'minute',
#         'second'
#     ]


# def month_year_new_features_names(function_transformer, feature_names_in):
#     return [
#         'expiration_month',
#         'expiration_year',
#         'acct_open_month',
#         'acct_open_year'
#     ]


# cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

cols_money_to_int = [
    "total_debt",
    "yearly_income",
    "per_capita_income",
    "credit_limit",
    "amount"
]

cols_month_year_date_to_int = [
    "acct_open_date",
    "expires"
]

cols_category_to_int = {
    "use_chip": ["Chip Transaction", "Online Transaction", "Swipe Transaction"],
    "gender": ["M", "F"],
    "card_brand": ["Visa", "Mastercard", "Discover", "Amex"],
    "card_type": ["Debit", "Debit (Prepaid)", "Credit"],
    "has_chip": ["YES", "NO"],
}

cols_to_lookup = [
    "merchant_city",
    "merchant_state"
]

cols_to_drop = [
    "errors",
    "card_on_dark_web",
    "address",
    "latitude",
    "longitude"
]


def build_one_hot_encoder(vocab):
    return tf.keras.Sequential([
        layers.StringLookup(
            vocabulary=vocab,
            output_mode="int"
        ),
        layers.CategoryEncoding(num_tokens=len(
            vocab) + 1,
            output_mode="one_hot"
        )
    ])


def build_bow_vectorizer(max_tokens=1000, ngrams=1):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        ngrams=ngrams,
        output_mode="count",
        standardize="lower_and_strip_punctuation",
        split="whitespace"
    )
    return vectorizer


vectorizer = build_bow_vectorizer()
description_ds = get_column_values(train_dataset, "description")
vectorizer.adapt(description_ds)


# ------------------------------------------------------------------------------
#   Transforming mercahnt city and merchant state into lookups
# ------------------------------------------------------------------------------
lookup_layers = {}
for col in cols_to_lookup:
    # Convert column to string (avoids issues with NaNs)
    # col_data = df[col].astype(str).values
    col_data = get_column_values(train_dataset, col)
    # Create lookup layer
    col_lookup = tf.keras.layers.StringLookup(output_mode='int')
    col_lookup.adapt(col_data)
    # lookup_layers[col] = col_lookup(df[col].astype(str).values).numpy()
    lookup_layers[col] = col_lookup

# ------------------------------------------------------------------------------
#   Preprocessing
# ------------------------------------------------------------------------------

encoders = {col: build_one_hot_encoder(
    vocab) for col, vocab in cols_category_to_int.items()}


def preprocess(features, label):
    for col_name, vocab in cols_category_to_int.items():
        features[col_name] = encoders[col_name](features[col_name])

    for col in cols_money_to_int:
        features[col] = dollar_to_float(
            features[col]
        )
    features["date"] = transaction_date_to_int(
        features["date"]
    )

    features["description"] = vectorizer(features["description"])

    for col in cols_month_year_date_to_int:
        features[col] = month_year_date_to_int(
            features[col]
        )

    for col in cols_to_lookup:
        # Transform into integer IDs
        features[col] = lookup_layers[col](features[col])

    for col_to_drop in cols_to_drop:
        if col_to_drop in features:
            del features[col_to_drop]

    return features, label


processed_dataset = (
    train_dataset
    .map(preprocess)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


def extract_columns_and_dtypes(spec, parent_key=''):
    columns = []
    for key, value in spec.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            columns.extend(extract_columns_and_dtypes(
                value, parent_key=full_key))
        else:
            columns.append((full_key, value.dtype))
    return columns


# Unpack features and label from dataset
features_spec, label_spec = processed_dataset.element_spec

# Now extract column info
columns_info = extract_columns_and_dtypes(features_spec)

# Print the results
for name, dtype in columns_info:
    print(f"{name}: {dtype}")


if __name__ == "__main__":
    features_batch, labels_batch = next(iter(train_dataset))

    # Aplica preprocessamento
    X, y = preprocess(features_batch, labels_batch)
