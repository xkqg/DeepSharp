# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes two models cut short, from keras-attention.keras with one number of its weights taken out:
#
#   keras-attention-missing.keras   the bias of the key projection
#   keras-attention-bare.keras      the kernel of the value projection
#
# The reader refuses each at the layer's projection and names the numbers it never found. The groups and datasets are copied in the order
# they stand in, with their attributes, since the reader finds a layer by where it stands.
#
# Run from this folder, after keras-attention-fixtures.py, with h5py:
#
#     python keras-attention-missing-fixture.py

import io
import os
import zipfile

import h5py

here = os.path.dirname(os.path.abspath(__file__))


def copy(source, target, path, leave, removed):
    for key, value in source.attrs.items():
        target.attrs[key] = value

    for name in source:
        item = source[name]
        where = f"{path}/{name}"

        if isinstance(item, h5py.Dataset):
            if leave(where):
                removed.append(where)
                continue

            target.create_dataset(name, data=item[()])
        else:
            copy(item, target.create_group(name, track_order=True), where, leave, removed)


def cut(name, leave):
    removed = []

    with zipfile.ZipFile(os.path.join(here, "keras-attention.keras")) as source, zipfile.ZipFile(os.path.join(here, name), "w") as target:
        for entry in source.infolist():
            data = source.read(entry.filename)

            if entry.filename == "model.weights.h5":
                kept = io.BytesIO()

                with h5py.File(io.BytesIO(data), "r", track_order=True) as read, h5py.File(kept, "w", track_order=True) as write:
                    copy(read, write, "", leave, removed)

                assert removed, "nothing to take out"
                data = kept.getvalue()

            target.writestr(entry, data)


cut("keras-attention-missing.keras", lambda where: "key_dense" in where and where.endswith("/1"))
cut("keras-attention-bare.keras", lambda where: "value_dense" in where and where.endswith("/0"))
