// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// Reads a model trained somewhere else into a network here: what a reader of one framework's files implements, the way an
/// engine implements <see cref="Tensors.ITensorBackend"/>.
/// </summary>
/// <remarks>
/// An importer hands back what reading a network's own file hands back — the network, every slot holding the numbers the
/// file holds, and the loss it answers through — so nothing downstream asks where a network came from. It is one of two
/// kinds. One reads numbers alone, from a file that names no layers, and is handed the network they belong to and its loss
/// when it is made; the other reads a file that describes the network too, and builds it from that description. Either
/// hands the numbers to <see cref="Network.Load"/> by the path of each slot, so every importer refuses what a network's own
/// file refuses, in the same words, each fault placed where its file holds it.
/// <para>
/// An import records no fit of a pipeline, so it stands behind one only as any network does: compiled with an optimizer and
/// its loss, and fitted here on the rows that pipeline prepares — for one epoch, or for as many as training it further
/// takes. That fit is what the network's one file records, beside the pipeline it was fitted behind.
/// </para>
/// </remarks>
public interface IImporter
{
    /// <summary>Reads a file of the kind this importer reads.</summary>
    /// <param name="file">The file, read from where it stands; left open.</param>
    /// <returns>The network, every slot holding the numbers the file holds, and the loss it answers through.</returns>
    /// <exception cref="SlotLoadException">The file's numbers do not fit the network: every fault at once, and no slot changed.</exception>
    /// <exception cref="FormatException">The file is not one of the kind this importer reads.</exception>
    SavedNetwork Read(Stream file);
}
