// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// What PyTorch makes of a convolution or a pooling along one, two or three axes, and of the gradient of a loss through it:
/// the values Fixtures/spatial-pytorch.py printed, for every engine to be held to. Everything is laid out with the channels
/// last, as the seam lays it out.
/// </summary>
internal sealed class SpatialReference
{
    private SpatialReference(JsonElement written)
    {
        Name = written.GetProperty("name").GetString()!;
        Kind = written.GetProperty("kind").GetString()!;
        Rank = written.GetProperty("rank").GetInt32();
        Count = written.GetProperty("n").GetInt32();
        Extents = Wholes(written, "extents");
        Channels = written.GetProperty("channels").GetInt32();
        Input = Values(written, "input");
        Output = Values(written, "output");
        OutShape = new Shape(Wholes(written, "outShape"));
        GradOutput = Values(written, "gradOutput");
        GradInput = Values(written, "gradInput");

        if (written.TryGetProperty("sizes", out var sizes))
        {
            Sizes = [.. sizes.EnumerateArray().Select(size => size.GetInt32())];
            Stride = written.GetProperty("stride").GetInt32();

            var padding = written.GetProperty("padding");

            if (padding.ValueKind == JsonValueKind.Number)
            {
                Padding = padding.GetInt32();
            }
            else
            {
                PaddingMode = padding.GetString() == "same" ? PaddingMode.Same : PaddingMode.Causal;
            }
        }

        if (written.TryGetProperty("outChannels", out var outChannels))
        {
            OutChannels = outChannels.GetInt32();
            Kernel = Values(written, "kernel");
            Bias = Values(written, "bias");
            GradKernel = Values(written, "gradKernel");
            GradBias = Values(written, "gradBias");
        }

        CountsPadding = written.TryGetProperty("countsPadding", out var counts) && counts.GetBoolean();
    }

    /// <summary>Every case the fixture holds.</summary>
    public static IReadOnlyList<SpatialReference> All { get; } = Read();

    public string Name { get; }

    /// <summary>What it is: <c>convolution</c>, <c>max</c>, <c>average</c>, <c>globalAverage</c> or <c>globalMax</c>.</summary>
    public string Kind { get; }

    /// <summary>How many axes the window walks: one, two or three.</summary>
    public int Rank { get; }

    public int Count { get; }

    public int[] Extents { get; }

    public int Channels { get; }

    public int OutChannels { get; }

    public int[] Sizes { get; } = [];

    public int Stride { get; }

    public int Padding { get; }

    public PaddingMode PaddingMode { get; }

    public bool CountsPadding { get; }

    public float[] Input { get; }

    public float[] Kernel { get; } = [];

    public float[] Bias { get; } = [];

    public float[] Output { get; }

    public Shape OutShape { get; }

    public float[] GradOutput { get; }

    public float[] GradInput { get; }

    public float[] GradKernel { get; } = [];

    public float[] GradBias { get; } = [];

    /// <summary>The names of the cases of one kind and rank, for a theory to run over.</summary>
    public static TheoryData<string> Named(string kind, params int[] ranks) =>
        [.. All.Where(reference => reference.Kind == kind && (ranks.Length == 0 || ranks.Contains(reference.Rank))).Select(reference => reference.Name)];

    /// <summary>The case of a name.</summary>
    public static SpatialReference Of(string name) => All.Single(reference => reference.Name == name);

    /// <summary>The images, volumes or series the case handed PyTorch, channels last.</summary>
    public Tensor Images() => Tensor.From(new Shape([Count, .. Extents, Channels]), Input);

    /// <summary>The gradient of the loss at the layer's output: what the case multiplied it by before adding it up.</summary>
    public Tensor Gradient() => Tensor.From(OutShape, GradOutput);

    public Window1D Along() => new(Sizes[0]) { Stride = Stride, Padding = Padding, PaddingMode = PaddingMode };

    public Window Over() => new(Sizes[0], Sizes[1]) { Stride = Stride, Padding = Padding, PaddingMode = PaddingMode };

    public Window3D Through() => new(Sizes[0], Sizes[1], Sizes[2]) { Stride = Stride, Padding = Padding, PaddingMode = PaddingMode };

    /// <summary>The kernel and bias PyTorch had, as the convolution of the case's rank.</summary>
    public Convolution Convolution()
    {
        var kernel = Tensor.From(new Shape(Kernel.Length / OutChannels, OutChannels), Kernel);
        var bias = Tensor.From(new Shape(OutChannels), Bias);

        return Rank switch
        {
            1 => new Conv1D(kernel, bias, Along()),
            2 => new Conv2D(kernel, bias, Over()),
            _ => new Conv3D(kernel, bias, Through()),
        };
    }

    /// <summary>The pooling of the case's kind, rank and window; a global pooling when the case holds no window.</summary>
    public Layer Pooling() => (Kind, Rank) switch
    {
        ("max", 1) => new MaxPool1D(Along()),
        ("max", 2) => new MaxPool2D(Over()),
        ("max", 3) => new MaxPool3D(Through()),
        ("average", 1) => new AvgPool1D(Along()) { CountsPadding = CountsPadding },
        ("average", 2) => new AvgPool2D(Over()) { CountsPadding = CountsPadding },
        ("average", 3) => new AvgPool3D(Through()) { CountsPadding = CountsPadding },
        ("globalAverage", 1) => new GlobalAvgPool1D(),
        ("globalAverage", 2) => new GlobalAvgPool2D(),
        ("globalAverage", 3) => new GlobalAvgPool3D(),
        ("globalMax", 1) => new GlobalMaxPool1D(),
        ("globalMax", 2) => new GlobalMaxPool2D(),
        ("globalMax", 3) => new GlobalMaxPool3D(),
        _ => throw new InvalidOperationException($"No layer pools as '{Kind}' along {Rank} axes."),
    };

    /// <summary>The sum of the layer's output times the gradient the case multiplied it by: the loss PyTorch took the gradient of.</summary>
    public Tensor LossOver(ITensorBackend backend, Tensor output) =>
        backend.Scale(backend.Mean(backend.Multiply(output, Gradient())), backend.Fill(new Shape(), output.Shape.Count));

    private static List<SpatialReference> Read()
    {
        var path = Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "spatial-pytorch.json");

        using var document = JsonDocument.Parse(File.ReadAllText(path));

        return [.. document.RootElement.GetProperty("cases").EnumerateArray().Select(written => new SpatialReference(written))];
    }

    private static int[] Wholes(JsonElement written, string name) => [.. written.GetProperty(name).EnumerateArray().Select(length => length.GetInt32())];

    private static float[] Values(JsonElement written, string name) =>
        [.. written.GetProperty(name).EnumerateArray().Select(value => (float)value.GetDouble())];

    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{Name}: {Count}x{string.Join('x', Extents)}x{Channels}");
}
