# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 21:58:23 2025

@author: PC
"""
from datetime import datetime
from tensorflow.keras import layers
import tensorflow as tf
from pathlib import Path
import pandas as pd
from helperFunctions import *
import numpy as np
import kagglehub


def load_financial_data():
    """
    Downloads and loads various financial datasets, including transactions, user,
    and card data, and merges them with fraud labels.

    The function performs the following steps:
    1.  Downloads the latest version of the transactions dataset from Kaggle Hub.
    2.  Loads transactions in chunks from the downloaded dataset.
    3.  Loads user and card data from local CSV files.
    4.  Loads fraud labels from a local JSON file.
    5.  Performs K-means clustering on user latitude and longitude data.
    6.  Adds the cluster labels to the users data.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value but processes and prepares
        data for further use.

    Notes
    -----
    - The transaction data is loaded in chunks to handle large files efficiently.
    - User and card data are loaded once into memory.
    - K-means clustering is used to group users based on their geographical location.
    - File paths are hardcoded for local datasets, except for the transactions
      data which is downloaded from Kaggle Hub.
    """
    # Download latest version
    path = kagglehub.dataset_download(
        "computingvictor/transactions-fraud-datasets"
    )

    print("Path to dataset files:", path)

    output_path = "data.csv"
    # Load transactions in chunks
    transactions_chunks = pd.read_csv(
        Path(path + "/transactions_data.csv"),
        chunksize=500_000
    )

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
        chunk = pd.merge(chunk, train_fraud_labels, on='id', how='left')
        chunk = chunk[chunk['label'].notna()]
        fraud_rows = chunk[chunk['label'] == "Yes"]
        nonfraud_rows = chunk[chunk['label'] == "No"]
        nonfraud_sample = nonfraud_rows.sample(
            n=min(len(nonfraud_rows),
                  len(fraud_rows) * 10
                  ),
            random_state=42
        )
        chunk = pd.concat([fraud_rows, nonfraud_sample]
                          ).sample(frac=1, random_state=42)
        chunk = pd.merge(
            chunk,
            cards_data,
            on=['client_id', 'card_id'],
            how='left'
        )
        chunk = pd.merge(chunk, users_data, on='client_id', how='left')
        mode = 'w' if first_chunk else 'a'
        chunk.to_csv(
            "data.csv",
            mode=mode,
            header=first_chunk,
            index=False
        )

        first_chunk = False  # Set the flag to False after the first write.

    print(f"All data successfully merged and saved to {output_path}.")
    return None


# load_financial_data()
csv_file = "data.csv"

dataset = tf.data.experimental.make_csv_dataset(
    file_pattern=csv_file,
    batch_size=32,
    label_name='label',
    num_epochs=1,
    shuffle=True,
    shuffle_seed=42,

    # shuffle_buffer_size=415809
)

# df = pd.read_csv(csv_file)
# print("Number of rows:", len(df))

dataset_size = 100000

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
    """
    Converts a dollar-formatted string tensor into a floating-point number tensor.

    This function removes the leading dollar sign '$' from each string in the
    input tensor and then converts the resulting string to a 32-bit
    floating-point number.

    Parameters
    ----------
    X : tf.Tensor
        A string tensor containing values in dollar format (e.g., "$123.45").

    Returns
    -------
    tf.Tensor
        A float32 tensor with the dollar signs removed and the values
        converted to numbers.

    Notes
    -----
    - Uses `tf.strings.regex_replace` to efficiently remove the '$' character.
    - Uses `tf.strings.to_number` to convert the cleaned strings to floats.
    """
    value = tf.strings.regex_replace(X, r'[$]', '')
    return tf.strings.to_number(value, tf.float32)


def transaction_date_to_int(X):
    """
    Parses a transaction date string tensor and extracts its components as integers.

    This function takes a date string in "YYYY-MM-DDTHH:MM:SS" format,
    extracts the year, month, day, hour, minute, and second, and returns
    them as a list of integer tensors.

    Parameters
    ----------
    X : tf.Tensor
        A string tensor containing transaction dates in ISO 8601 format.

    Returns
    -------
    list
        A list of six `tf.Tensor` objects, each containing integer values for
        year, month, day, hour, minute, and second, respectively.

    Notes
    -----
    - Uses `tf.strings.substr` to extract specific parts of the date string.
    - Uses `tf.strings.to_number` to convert the string parts to integers.
    """
    year = tf.strings.substr(X, 0, 4)
    month = tf.strings.substr(X, 5, 2)
    day = tf.strings.substr(X, 8, 2)
    hour = tf.strings.substr(X, 11, 2)
    minute = tf.strings.substr(X, 14, 2)
    second = tf.strings.substr(X, 17, 2)

    # Convert to numbers
    return [
        tf.strings.to_number(year, tf.int32),
        tf.strings.to_number(month, tf.int32),
        tf.strings.to_number(day, tf.int32),
        tf.strings.to_number(hour, tf.int32),
        tf.strings.to_number(minute, tf.int32),
        tf.strings.to_number(second, tf.int32),
    ]


def month_year_date_to_int(X):
    """
    Parses a month/year date string tensor and extracts its components as integers.

    This function takes a date string in "MM-YY" format, extracts the month
    and year, and returns them as a list of integer tensors.

    Parameters
    ----------
    X : tf.Tensor
        A string tensor containing dates in "MM-YY" format.

    Returns
    -------
    list
        A list of two `tf.Tensor` objects, each containing integer values for
        month and year, respectively.

    Notes
    -----
    - Uses `tf.strings.substr` to extract the month and year parts.
    - Uses `tf.strings.to_number` to convert the string parts to integers.
    """
    month = tf.strings.substr(X, 0, 2)
    year = tf.strings.substr(X, 3, 2)
    # Convert to numbers
    return [
        tf.strings.to_number(month, tf.int32),
        tf.strings.to_number(year, tf.int32),
    ]


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

cols_dont_need_preprocessing = [
    "zip",
    "num_cards_issued",
    "birth_year",
    "birth_month",
    "num_credit_cards",
    "credit_score",
]


def build_one_hot_encoder(vocab):
    """
    Builds a Keras `CategoryEncoding` layer for one-hot encoding categorical features.

    Parameters
    ----------
    vocab : list
        A list of unique vocabulary items (e.g., strings) to be encoded.

    Returns
    -------
    tf.keras.layers.Layer
        A configured `CategoryEncoding` layer that can be used to convert
        integer-encoded categorical data into one-hot vectors.

    Notes
    -----
    - The `num_tokens` parameter is set to `len(vocab) + 1` to account for
      the zero-padding value, which is often used for out-of-vocabulary
      items or missing values.
    - `output_mode="one_hot"` specifies that the layer should produce
      a one-hot encoded output, where each input integer is mapped to a
      vector of zeros with a single '1' at the corresponding index.
    """
    return layers.CategoryEncoding(num_tokens=len(
        vocab) + 1,
        output_mode="one_hot"
    )


# ------------------------------------------------------------------------------
#   Preprocessing
# ------------------------------------------------------------------------------
lookups = {
    col: tf.keras.layers.StringLookup(vocabulary=vocab, output_mode="int")
    for col, vocab in cols_category_to_int.items()
}
encoders = {
    col: build_one_hot_encoder(
        vocab) for col, vocab in cols_category_to_int.items()}


def preprocess(features, label):
    """
    Preprocesses a batch of financial transaction data for a machine learning model.

    This function transforms raw features into a numerical format suitable for model training.
    It handles categorical, monetary, and date-based features, and also calculates
    the years since the last PIN change.

    Parameters
    ----------
    features : dict
        A dictionary of tensors where keys are feature names and values are
        the corresponding feature data for a batch.
    label : tf.Tensor
        A tensor containing the labels (e.g., "Yes" or "No" for fraud).

    Returns
    -------
    tuple
        A tuple containing three tensors:
        - X_processed (tf.Tensor): The concatenated and processed features.
        - label_processed (tf.Tensor): The processed labels (0 for "No", 1 for "Yes").
        - one_hot (tf.Tensor): The one-hot encoded categorical features.

    Notes
    -----
    - **Categorical features**: Mapped to integer IDs and then converted to one-hot vectors.
    - **Monetary features**: Converted from string representations to floating-point numbers.
    - **Date features**: Processed to extract numerical components (e.g., day, month, year).
    - **years_since_pin_change**: Calculated by subtracting the 'year_pin_last_changed'
      from the current year.
    - The final output for features (X_processed) is a single concatenated tensor.
    """
    features_flattened = []
    one_hot_parts = []
    feature_names = []

    for col, vocab in cols_category_to_int.items():
        # Get the prebuilt lookup & encoder for this column
        col_lookup = lookups[col]
        col_encoder = encoders[col]

        # Map feature column -> integer IDs -> one-hot vector
        ids = col_lookup(features[col])
        enc_out = col_encoder(ids)

        # Store the one-hot as float
        one_hot_parts.append(tf.cast(enc_out, tf.float32))
        feature_names.append(col)

        # Concatenate all one-hot outputs for this example
        one_hot = tf.concat(one_hot_parts, axis=1)

    for col in cols_money_to_int:
        val = dollar_to_float(features[col])
        features_flattened.append(tf.expand_dims(val, axis=-1))
        feature_names.append(col)

    features_flattened.extend([
        tf.expand_dims(tf.cast(part, tf.float32), axis=-1)
        for part in transaction_date_to_int(features["date"])
    ])

    for col in cols_month_year_date_to_int:
        features_flattened.extend([
            tf.expand_dims(tf.cast(part, tf.float32), axis=-1)
            for part in month_year_date_to_int(features[col])
        ])
        feature_names.append(col)

    for col in cols_dont_need_preprocessing:
        val = tf.cast(features[col], tf.float32)
        features_flattened.append(tf.expand_dims(val, axis=-1))
        feature_names.append(col)

    # Subtract from current year
    current_year = tf.constant(datetime.now().year, dtype=tf.int32)
    years_since_pin_change = tf.cast(
        current_year - features["year_pin_last_changed"], tf.float32)

    # Cast to float32 after subtraction
    years_since_pin_change = tf.cast(years_since_pin_change, tf.float32)

    years_since_pin_change = tf.expand_dims(years_since_pin_change, axis=-1)
    # Add to features_flattened
    features_flattened.append(years_since_pin_change)
    # print(feature_names)

    X_processed = tf.concat(features_flattened, axis=1)

    label_processed = tf.where(tf.equal(label, "No"), 0, 1)
    label_processed = tf.cast(label_processed, tf.float32)
    return X_processed, label_processed, one_hot


def preprocess_data():
    """
    Applies the `preprocess` function to training and validation datasets
    and concatenates the results.

    This function iterates through the provided training and validation datasets,
    applies the `preprocess` function to each batch, and then concatenates
    all processed batches into single NumPy arrays for each feature set.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        A tuple of six NumPy arrays:
        - train_X (np.ndarray): Processed features for the training set.
        - train_Y (np.ndarray): Processed labels for the training set.
        - val_X (np.ndarray): Processed features for the validation set.
        - val_Y (np.ndarray): Processed labels for the validation set.
        - train_one_hot (np.ndarray): One-hot encoded categorical features for the training set.
        - val_one_hot (np.ndarray): One-hot encoded categorical features for the validation set.

    Notes
    -----
    - The function assumes that `train_dataset` and `val_dataset` are already defined
      and iterable (e.g., `tf.data.Dataset` objects).
    - It concatenates the output of each batch-wise `preprocess` call to form
      final, single arrays for both training and validation data.
    """
    # Training data
    train_X, train_Y, train_one_hot = [], [], []

    for features_batch, labels_batch in train_dataset:
        X, Y, one_hot_batch = preprocess(features_batch, labels_batch)
        train_X.append(X)
        train_Y.append(Y)
        train_one_hot.append(one_hot_batch)

    train_X = np.concatenate(train_X, axis=0)
    train_Y = np.concatenate(train_Y, axis=0)
    train_one_hot = np.concatenate(train_one_hot, axis=0)

    # Validation data
    val_X, val_Y, val_one_hot_list = [], [], []
    for features_batch, labels_batch in val_dataset:
        X, Y, val_one_hot_batch = preprocess(features_batch, labels_batch)
        val_X.append(X)
        val_Y.append(Y)
        val_one_hot_list.append(val_one_hot_batch)

    val_X = np.concatenate(val_X, axis=0)
    val_Y = np.concatenate(val_Y, axis=0)
    val_one_hot = np.concatenate(val_one_hot_list, axis=0)

    return train_X, train_Y, val_X, val_Y, train_one_hot, val_one_hot


# preprocess_data()
# load_financial_data()
