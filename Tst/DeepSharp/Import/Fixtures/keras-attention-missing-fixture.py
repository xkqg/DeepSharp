# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes keras-attention-missing.keras: keras-attention.keras with the bias of the key projection taken out of its weights, the file of a
# model that was cut short. The reader refuses it at the layer's projection and names the numbers it never found.
#
# Run from this folder, after keras-attention-fixtures.py, with h5py:
#
#     python keras-attention-missing-fixture.py

import io
import os
import zipfile

import h5py

here = os.path.dirname(os.path.abspath(__file__))

with zipfile.ZipFile(os.path.join(here, "keras-attention.keras")) as source, zipfile.ZipFile(os.path.join(here, "keras-attention-missing.keras"), "w") as target:
    for entry in source.infolist():
        data = source.read(entry.filename)

        if entry.filename == "model.weights.h5":
            with h5py.File(io.BytesIO(data), "r") as read:
                kept = io.BytesIO()
                removed = []

                with h5py.File(kept, "w") as write:
                    def copy(name, item):
                        if isinstance(item, h5py.Dataset):
                            if "key_dense" in name and name.endswith("/1"):
                                removed.append(name)
                                return
                            write.create_dataset(name, data=item[()])
                        elif name not in write:
                            write.require_group(name)

                    read.visititems(copy)
                    for key, value in read.attrs.items():
                        write.attrs[key] = value

                assert removed, "no bias of the key projection to take out"
                data = kept.getvalue()

        target.writestr(entry, data)
