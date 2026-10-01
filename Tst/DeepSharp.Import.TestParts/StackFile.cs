// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import.Parts;

/// <summary>
/// A reader of a file that describes the network too, as a framework of somebody else's saves a whole model: a line for
/// each layer of a stack — <c>dense 2 2</c>, <c>batchNorm 2</c>, <c>relu</c> — one for its loss, and the numbers as
/// <see cref="WeightsFile"/> holds them. It builds the stack it describes from the layers' own constructors and puts the
/// numbers in through the same public door.
/// </summary>
public sealed class StackFile : IImporter
{
    /// <inheritdoc />
    /// <exception cref="FormatException">The file names no loss.</exception>
    public SavedNetwork Read(Stream file)
    {
        var layers = new List<Layer>();
        var entries = new List<SlotEntry>();
        Loss? loss = null;

        foreach (var line in file.Lines())
        {
            switch (line.Words[0])
            {
                case "dense":
                    layers.Add(new Dense(Tensor.Zeros(new Shape(line.Whole(1), line.Whole(2))), Tensor.Zeros(new Shape(line.Whole(2)))));
                    break;
                case "batchNorm":
                    layers.Add(new BatchNorm(line.Whole(1)));
                    break;
                case "relu":
                    layers.Add(new Relu());
                    break;
                case "loss":
                    loss = line.Words[1] == "binaryCrossEntropy" ? new BinaryCrossEntropy() : new MeanSquaredError();
                    break;
                default:
                    entries.Add(line.Entry());
                    break;
            }
        }

        var network = new LayerStack(layers);
        network.Load(entries);

        return new SavedNetwork(network, loss ?? throw new FormatException("A file that describes its network names the loss it answers through."));
    }
}
