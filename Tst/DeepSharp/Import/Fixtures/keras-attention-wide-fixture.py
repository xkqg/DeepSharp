# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes one Keras model that no network here holds, in both files Keras 3 saves a model to: multi-head attention whose heads each ask in
# more channels than the step is split into (2 heads of key_dim 8 over steps of 8 channels, where a head here asks in 4). It is the form
# Keras's own text-classification example uses (key_dim as wide as the step), and the reader refuses it at the layer's slot.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-attention-wide-fixture.py

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261024)
rng = np.random.default_rng(20261024)

inputs = keras.Input(shape=(5, 8))
looked = layers.MultiHeadAttention(num_heads=2, key_dim=8)(inputs, inputs)
model = keras.Model(inputs, layers.Dense(2)(layers.GlobalAveragePooling1D()(looked)))
model.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
model.fit(rng.normal(size=(48, 5, 8)).astype(np.float32), rng.normal(size=(48, 2)).astype(np.float32), epochs=3, batch_size=16, verbose=0)
model.save(os.path.join(here, "keras-attention-wide.keras"))
model.save(os.path.join(here, "keras-attention-wide.h5"))
