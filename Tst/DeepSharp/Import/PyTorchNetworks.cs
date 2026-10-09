// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Learners;
using Onnxify.Safetensors;

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

    /// <summary>
    /// A network pytorch-spatial.py saved, in Keras's words, its layers numbered as PyTorch's Sequential numbers its modules, and
    /// the shape of one example it takes: <c>series</c>, <c>series-flat</c>, <c>pooling</c>, <c>pooling-global</c>,
    /// <c>volume</c> and <c>volume-global</c>. PyTorch's average pooling counts the cells it pads with, so these do.
    /// </summary>
    public static SpatialCase Spatial(string name) => name switch
    {
        "series" => new(() => SeriesInKerasWords(), new Shape(17, 2)),
        "series-flat" => new(() => SeriesFlatInKerasWords(), new Shape(10, 2)),
        "pooling" => new(
            () => new Sequential().Conv2D(4, new Window(3, 2) { Padding = 1 }).BatchNorm().Relu()
                .MaxPool2D(new Window(3, 2) { Stride = 2, Padding = 1 })
                .Conv2D(3, new Window(2, 2))
                .AvgPool2D(new Window(2, 2) { Stride = 1, Padding = 1 }, countsPadding: true)
                .SpatialDropout2D(0.2).Flatten().Dense(5).Tanh().Dense(1)
                .Lower(new Shape(8, 6, 2), new RandomStream(7)),
            new Shape(8, 6, 2)),
        "pooling-global" => new(
            () => new Sequential().Conv2D(4, new Window(3, 3) { Padding = 1 }).Relu()
                .GlobalMaxPool2D(keepsAxes: true).Flatten().Dense(3)
                .Lower(new Shape(6, 5, 2), new RandomStream(7)),
            new Shape(6, 5, 2)),
        "volume" => new(
            () => new Sequential().Conv3D(3, new Window3D(2, 3, 2) { Padding = 1 }).BatchNorm().Relu()
                .MaxPool3D(new Window3D(2, 2, 2) { Stride = 2, Padding = 1 })
                .Conv3D(2, new Window3D(2, 2, 2))
                .AvgPool3D(new Window3D(2, 2, 2) { Stride = 1, Padding = 1 }, countsPadding: true)
                .SpatialDropout3D(0.3).Flatten().Dense(3)
                .Lower(new Shape(5, 6, 4, 2), new RandomStream(7)),
            new Shape(5, 6, 4, 2)),
        _ => new(
            () => new Sequential().Conv3D(3, new Window3D(2, 2, 2)).Relu()
                .GlobalAvgPool3D(keepsAxes: true).Flatten().Dense(2)
                .Lower(new Shape(5, 6, 4, 2), new RandomStream(7)),
            new Shape(5, 6, 4, 2)),
    };

    /// <summary>pytorch-spatial.py's first series network in Keras's words, its adaptive average pooling written as a global one.</summary>
    /// <param name="keepsAxes">Whether the pooling keeps the axis it pools, as PyTorch's does; the flatten after it then has one place to flatten.</param>
    public static LayerStack SeriesInKerasWords(bool keepsAxes = true) =>
        new Sequential().Conv1D(4, new Window1D(3) { Stride = 2, Padding = 1 }).BatchNorm().Relu()
            .MaxPool1D(new Window1D(3) { Stride = 2, Padding = 1 })
            .Conv1D(5, new Window1D(2)).SpatialDropout1D(0.25)
            .GlobalAvgPool1D(keepsAxes).Flatten().Dense(3)
            .Lower(new Shape(17, 2), new RandomStream(7));

    /// <summary>pytorch-spatial.py's second series network in Keras's words.</summary>
    /// <param name="countsPadding">Whether the average pooling counts the cells it pads with, as PyTorch's does unless told not to.</param>
    public static LayerStack SeriesFlatInKerasWords(bool countsPadding = true) =>
        new Sequential().Conv1D(4, new Window1D(4) { PaddingMode = PaddingMode.Same }).Relu()
            .AvgPool1D(new Window1D(2) { Stride = 2, Padding = 1 }, countsPadding)
            .Conv1D(3, new Window1D(2)).Flatten().BatchNorm().Dense(4).Tanh().Dense(1)
            .Lower(new Shape(10, 2), new RandomStream(7));

    /// <summary>The examples pytorch-spatial.py handed one of its networks, in this library's layout.</summary>
    public static Tensor SpatialExamples(string name)
    {
        var examples = PyTorchFixture.Spatial.GetProperty(name).GetProperty("examples");

        return Tensor.From(new Shape([.. examples.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), examples.GetProperty("values").Floats());
    }

    /// <summary>What PyTorch answered through one of pytorch-spatial.py's networks.</summary>
    public static float[] SpatialOutputs(string name) => PyTorchFixture.Spatial.GetProperty(name).GetProperty("outputs").Floats();

    /// <summary>How far apart, at the most, two lists of answers lie.</summary>
    public static float Farthest(float[] ours, float[] theirs) => ours.Zip(theirs, (mine, torch) => Math.Abs(mine - torch)).Max();

    /// <summary>A safetensors file with each of its tensors under another name.</summary>
    public static MemoryStream Renamed(string file, Func<string, string> name)
    {
        var archive = SafeTensors.Deserialize(File.ReadAllBytes(PyTorchFixture.Path(file)));

        return new MemoryStream(SafeTensors.Serialize(archive.Tensors().Select(pair => new KeyValuePair<string, TensorView>(name(pair.Key), pair.Value)), archive.Metadata.MetadataEntries));
    }

    /// <summary>A safetensors file of the given tensors, written by the borrowed writer.</summary>
    public static MemoryStream Written(Held[] tensors, IReadOnlyDictionary<string, string>? notes = null) =>
        new(SafeTensors.Serialize(tensors.Select(held => new KeyValuePair<string, TensorView>(held.Name, new TensorView(held.Type, held.Shape, held.Bytes))), notes));

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

/// <summary>A network of pytorch-spatial.py written here, and the shape of one example it takes.</summary>
/// <param name="Network">Writes the network afresh.</param>
/// <param name="Example">One example, without the batch's axis.</param>
internal readonly record struct SpatialCase(Func<LayerStack> Network, Shape Example);

/// <summary>A tensor to be written into a safetensors file.</summary>
/// <param name="Name">Its name.</param>
/// <param name="Type">Its kind of number.</param>
/// <param name="Shape">Its shape.</param>
/// <param name="Bytes">Its bytes.</param>
internal readonly record struct Held(string Name, DataType Type, ulong[] Shape, byte[] Bytes);
