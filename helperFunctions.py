# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:49:35 2025

Author: PC

Description:
------------
This script provides utility functions for:
1. Performing K-means clustering on latitude and longitude coordinates.
2. Calculating the size of a dataset (with a normalization factor).
3. Extracting values from a specified column in a TensorFlow dataset.

Dependencies:
-------------
- numpy
- scikit-learn (sklearn.cluster.KMeans)
"""
import numpy as np
from sklearn.cluster import KMeans


def calcule_kmeans(latitudes, longitudes):
    """
    Performs K-means clustering on a given set of latitude and longitude coordinates.

    Parameters
    ----------
    latitudes : list or numpy.ndarray
        A list/array of latitude values.
    longitudes : list or numpy.ndarray
        A list/array of longitude values.

    Returns
    -------
    numpy.ndarray
        An array of cluster labels (integers from 0 to k-1), where each label
        corresponds to a point in the input coordinates.

    Notes
    -----
    - The number of clusters (k) is fixed at 10 in this implementation.
    - Coordinates are stacked into a 2D array before clustering.
    - The KMeans random_state ensures reproducibility.
    """
    # Convert lists to numpy arrays
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    # Stack them to create a 2D array
    coordinates = np.stack((latitudes, longitudes), axis=1)
    k = 10
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)

    cluster_labels = kmeans.fit_predict(coordinates)

    # Return the generated cluster labels
    return cluster_labels


def get_dataset_size(dataset):
    """
    Calculates the size of a dataset, normalized by a factor of 32.

    Parameters
    ----------
    dataset : iterable
        A dataset (e.g., TensorFlow Dataset object) where each iteration
        yields a batch of data.

    Returns
    -------
    float
        The size of the dataset divided by 32.

    Notes
    -----
    - This function assumes that each iteration of the dataset yields 1 item.
    - Dividing by 32 is related to batch size normalization.
    """
    count = 0
    for _ in dataset:
        count += 1
    return count/32


def get_column_values(train_dataset, column):
    """
    Extracts the values of a specified column from the first batch of a dataset.

    Parameters
    ----------
    train_dataset : tf.data.Dataset
        A TensorFlow dataset that yields tuples (features, labels).
        'features' is expected to be a dictionary-like object.
    column : str
        The column key whose values should be extracted.

    Returns
    -------
    list
        A list containing the values from the specified column.

    Notes
    -----
    - Byte strings are decoded to UTF-8 strings for readability.
    - Non-string data is converted to a Python list.
    """
    values = []

    for features, _ in train_dataset:
        col_values = features[column].numpy()
        if col_values.dtype.type is np.bytes_:
            col_values = [v.decode("utf-8") for v in col_values]
        else:
            col_values = col_values.tolist()
        values.extend(col_values)
        # if len(values) >= 10:
        #     break
    return values


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
