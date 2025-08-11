# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:44:50 2025

@author: PC
"""

import tensorflow as tf
from pathlib import Path
from time import strftime
import keras_tuner as kt
from model import preprocess

tf.random.set_seed(42)
X_train, Y_train, X_val, Y_val = preprocess()

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X_train)


def build_model(hp, normalizer):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-4,
        max_value=1e-2,
        sampling="log"
    )
    optimizer_name = hp.Choice("optimizer", values=[
        "sgd", "adamW", "momentum", "nesterov", "RMSProp"])
    match optimizer_name:
        case "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        case "adamW":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
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
    model.add(tf.keras.layers.Flatten())
    model.add(normalizer)
    model.add(tf.keras.layers.BatchNormalization())
    for _ in range(n_hidden):
        if (optimizer_name == "adamW"):
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
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_search_tuner.search(X_train, Y_train, X_val, Y_val, epochs=10)

top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]


def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir)/strftime("run_%Y_%m_%d_%H_%M_%S")


run_logdir = get_run_logdir()

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    run_logdir,
    profile_batch=(100, 200)
)

# history = model.fit([...], callbacks=[tensorboard_cb])
