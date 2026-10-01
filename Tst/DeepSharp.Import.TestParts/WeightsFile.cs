// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tests.Import.Parts;

/// <summary>
/// A reader of a file that holds numbers alone, as a framework of somebody else's saves a model's weights: a line for each
/// slot — its path, its shape and its values, <c>1.weight 2 1.5 0.5</c>. It names no layers, so it is handed the network the
/// numbers belong to and the loss it answers through, and puts the numbers in through the one public door.
/// </summary>
/// <param name="network">The network the numbers belong to.</param>
/// <param name="loss">What it answers through.</param>
public sealed class WeightsFile(Network network, Loss loss) : IImporter
{
    /// <inheritdoc />
    public SavedNetwork Read(Stream file)
    {
        network.Load([.. file.Lines().Select(line => line.Entry())]);

        return new SavedNetwork(network, loss);
    }
}
