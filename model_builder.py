"""utils/model_builder.py — ANN and 1D-CNN builders."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


OPTIMIZER_MAP = {
    "adam":    lambda lr: keras.optimizers.Adam(learning_rate=lr),
    "sgd":     lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    "rmsprop": lambda lr: keras.optimizers.RMSprop(learning_rate=lr),
    "adagrad": lambda lr: keras.optimizers.Adagrad(learning_rate=lr),
}


def _activation_layer(name: str, layer_id: int):
    act = name.lower()
    if act == "relu":       return layers.ReLU(name=f"relu_{layer_id}")
    if act == "leakyrelu":  return layers.LeakyReLU(alpha=0.1, name=f"lrelu_{layer_id}")
    if act == "elu":        return layers.ELU(name=f"elu_{layer_id}")
    if act == "tanh":       return layers.Activation("tanh", name=f"tanh_{layer_id}")
    return layers.ReLU(name=f"relu_{layer_id}")


def build_ann(
    n_features: int,
    activation: str = "relu",
    optimizer: str = "adam",
    dropout_rate: float = 0.3,
    l2_lambda: float = 1e-3,
    learning_rate: float = 1e-3,
):
    inp = keras.Input(shape=(n_features,), name="ann_input")

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(l2_lambda), name="d1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = _activation_layer(activation, 1)(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(l2_lambda), name="d2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = _activation_layer(activation, 2)(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda), name="d3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="act3")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop3")(x)

    out = layers.Dense(1, activation="sigmoid", name="ann_out")(x)

    opt = OPTIMIZER_MAP.get(optimizer.lower(), OPTIMIZER_MAP["adam"])(learning_rate)
    model = keras.Model(inp, out, name=f"ANN_{activation}_{optimizer}")
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def build_cnn1d(
    n_features: int,
    activation: str = "relu",
    optimizer: str = "adam",
    dropout_rate: float = 0.3,
    l2_lambda: float = 1e-3,
    learning_rate: float = 1e-3,
):
    act_str = {"relu": "relu", "leakyrelu": "relu", "elu": "elu", "tanh": "tanh"}.get(
        activation.lower(), "relu"
    )
    inp = keras.Input(shape=(n_features, 1), name="cnn_input")

    x = layers.Conv1D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_lambda), name="c1")(inp)
    x = layers.BatchNormalization(name="cbn1")(x)
    x = layers.Activation(act_str, name="cact1")(x)
    x = layers.Dropout(dropout_rate, name="cdrop1")(x)

    x = layers.Conv1D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_lambda), name="c2")(x)
    x = layers.BatchNormalization(name="cbn2")(x)
    x = layers.Activation(act_str, name="cact2")(x)
    x = layers.Dropout(dropout_rate, name="cdrop2")(x)

    x = layers.Conv1D(64, 3, padding="same", name="c3")(x)
    x = layers.BatchNormalization(name="cbn3")(x)
    x = layers.Activation(act_str, name="cact3")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda), name="dhead")(x)
    x = layers.Dropout(dropout_rate / 2, name="ddrop")(x)
    out = layers.Dense(1, activation="sigmoid", name="cnn_out")(x)

    opt = OPTIMIZER_MAP.get(optimizer.lower(), OPTIMIZER_MAP["adam"])(learning_rate)
    model = keras.Model(inp, out, name=f"CNN1D_{activation}_{optimizer}")
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model
