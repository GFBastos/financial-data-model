# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:44:50 2025

@author: PC
"""

from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from pathlib import Path
from time import strftime
import keras_tuner as kt
from loading_data_preprocessing import preprocess_data
import numpy as np
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.utils import class_weight
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW

tf.random.set_seed(42)
X_train, Y_train, X_val, Y_val, one_hot, val_one_hot = preprocess_data()
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X_train)
X_train_num = normalizer(tf.convert_to_tensor(X_train, dtype=tf.float32))
X_val_num = normalizer(tf.convert_to_tensor(X_val, dtype=tf.float32))

# Before concatenation
X_train = tf.concat(
    [X_train_num, tf.convert_to_tensor(one_hot, dtype=tf.float32)],
    axis=1
)
X_val = tf.concat(
    [X_val_num, tf.convert_to_tensor(val_one_hot, dtype=tf.float32)],
    axis=1
)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y_train),
    y=Y_train
)

class_weights = dict(zip(np.unique(Y_train), weights))


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train.numpy(), Y_train)

# -----------------------------------------------------------------------------
# print("Baseline AUC:", roc_auc_score(
#     Y_val, clf.predict_proba(X_val.numpy())[:, 1]))
# -----------------------------------------------------------------------------


def build_model(hp):
    """
    Builds and compiles a Keras Sequential model for a hyperparameter tuning task.

    The function dynamically constructs a neural network with configurable
    hyperparameters, including the number of hidden layers, the number of
    neurons per layer, the learning rate, and the optimizer.

    Parameters
    ----------
    hp : keras_tuner.HyperParameters
        An object that provides the range and choices for hyperparameters.

    Returns
    -------
    tf.keras.Sequential
        A compiled Keras Sequential model ready for training.

    Notes
    -----
    - **Hyperparameter Tuning**: The model's architecture is defined by
      `hp.Int`, `hp.Float`, and `hp.Choice`, allowing a hyperparameter tuner
      (like KerasTuner) to explore different configurations.
    - **Architecture**: The model consists of a series of hidden layers,
      each followed by a `BatchNormalization` layer. A `Dense` layer with a
      `relu` activation and `he_normal` kernel initializer is used for the
      hidden layers.
    - **Regularization**: `L2` regularization is applied to the kernel of
      dense layers, except when the "adamw" optimizer is chosen, as `AdamW`
      already handles weight decay.
    - **Output Layer**: The final layer is a single neuron with a `sigmoid`
      activation, suitable for binary classification.
    - **Compilation**: The model is compiled with `binary_crossentropy`
      loss and a configurable optimizer. It also includes several metrics
      (`accuracy`, `Precision`, `Recall`, and `AUC`) for performance evaluation.
    """
    n_hidden = hp.Int("n_hidden", min_value=3, max_value=6, default=4)
    n_neurons = hp.Int("n_neurons", min_value=128, max_value=1024, step=128)
    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-4,
        max_value=1e-2,
        sampling="log"
    )
    # learning_rate = 1e-3
    optimizer_name = hp.Choice("optimizer", values=[
        "sgd", "adamw", "momentum", "nesterov", "RMSProp"])
    match optimizer_name:
        case "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
        case "adamw":
            optimizer = AdamW(learning_rate=learning_rate)
        case "momentum":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9)
        case "nesterov":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9, nesterov=True)
        case "RMSProp":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                rho=0.9
            )

    model = tf.keras.Sequential()
    for _ in range(n_hidden):
        model.add(tf.keras.layers.BatchNormalization())
        if (optimizer_name == "adamw"):
            model.add(tf.keras.layers.Dense(
                n_neurons,
                activation="relu",
                kernel_initializer="he_normal"
            ))
        else:
            model.add(tf.keras.layers.Dense(
                n_neurons,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            ))
        # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", Precision(), Recall(), AUC()]
    )
    return model


random_search_tuner = kt.RandomSearch(
    build_model, objective="val_auc", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)

random_search_tuner.search(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=10,
    class_weight=class_weights
)

# Get best hyperparameters from the tuner
best_hps = random_search_tuner.get_best_hyperparameters(num_trials=1)[0]

# Build a fresh model from scratch using best_hps
fresh_best_model = build_model(best_hps)


def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir)/strftime("run_%Y_%m_%d_%H_%M_%S")


run_logdir = get_run_logdir()

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    run_logdir,
    profile_batch=(100, 200)
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True
)

history = fresh_best_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    callbacks=[tensorboard_cb, early_stopping_cb],
    class_weight=class_weights
)

fresh_best_model.save("best_model.keras")
