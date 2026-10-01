// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The networks PyTorch's fixtures were saved from, written here, the rows they were handed, and how far apart two answers
/// or two sets of numbers lie: what every reader of a file PyTorch saved is held to.
/// </summary>
internal static class PyTorchNetworks
{
    /// <summary>The engine every answer here is worked out on.</summary>
    public static readonly ITensorBackend Backend = new CpuBackend();

    /// <summary>The README's passenger, a man of 22 in third class, and a woman of 38 in first.</summary>
    public static readonly InMemoryRowSource Passengers = new(
        ["pclass", "sex", "age", "sibsp", "parch", "fare"],
        [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]);

    /// <summary>The pipeline the Titanic networks were trained behind, run over the wiki's passenger list.</summary>
    public static readonly Lazy<PreparedData> Titanic = new(() => WikiTitanic.In(WikiTitanic.DataFolder).Run());

    /// <summary>Titanic's network in Keras's words: sixteen, the rectifier, one; its layers numbered as PyTorch's Sequential numbers them.</summary>
    public static LayerStack TitanicInKerasWords() => new Sequential().Dense(16).Relu().Dense(1).Lower(new Shape(14), new RandomStream(7));

    /// <summary>pytorch.py's convolutions in Keras's words, the flatten written as a flatten or as a reshape into rows of 36.</summary>
    public static LayerStack ConvolutionInKerasWords(bool reshapedIntoRows = false)
    {
        var words = new Sequential().Conv2D(4, new Window(3, 2) { Padding = 1 }).BatchNorm().Relu().Conv2D(3, new Window(2, 2) { Stride = 2 });

        words = reshapedIntoRows ? words.Reshape(new Shape(36)) : words.Flatten();

        return words.BatchNorm().Dense(5).Tanh().Dense(1).Lower(new Shape(8, 6, 2), new RandomStream(7));
    }

    /// <summary>The six images pytorch.py handed its convolutions, in this library's layout.</summary>
    public static Tensor Images()
    {
        var images = PyTorchFixture.Json.GetProperty("convolution").GetProperty("images");

        return Tensor.From(new Shape([.. images.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), images.GetProperty("values").Floats());
    }

    /// <summary>Rows a pipeline prepared, as the tensor a network takes.</summary>
    public static Tensor Rows(IReadOnlyList<double[]> rows) =>
        Tensor.From(new Shape(rows.Count, rows[0].Length), [.. rows.SelectMany(row => row.Select(value => (float)value))]);

    /// <summary>How many roundings apart each chance here lies from PyTorch's, taken to thirty-two bits.</summary>
    public static IEnumerable<long> Roundings(float[] ours, double[] theirs) =>
        ours.Zip(theirs, (mine, torch) => Math.Abs((long)BitConverter.SingleToInt32Bits(mine) - BitConverter.SingleToInt32Bits((float)torch)));

    /// <summary>What every slot of a network holds, in the order it lists them.</summary>
    public static float[][] Values(Network network) => [.. network.Slots().Select(named => named.Slot.Value.Values.ToArray())];

    /// <summary>The bits of each number.</summary>
    public static int[] Bits(float[] values) => [.. values.Select(BitConverter.SingleToInt32Bits)];
}
