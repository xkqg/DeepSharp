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
/// <remarks>
/// A stride and a dilation are read as a list, one number an axis, whether or not the axes agree; a padding is read as the
/// one number on every side, as the word for a border worked out, or as a flat pair for each axis where each end is stated
/// apart. What a case does not say it does not have: no dilation is a dilation of one, no groups is one group, and a case
/// that says nothing of rounding stands only where its window fits whole.
/// </remarks>
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
        Groups = written.TryGetProperty("groups", out var groups) ? groups.GetInt32() : 1;
        RoundsUp = written.TryGetProperty("roundsUp", out var rounds) && rounds.GetBoolean();
        Value = written.TryGetProperty("value", out var value) ? (float)value.GetDouble() : 0f;

        if (written.TryGetProperty("outSizes", out var outSizes))
        {
            OutSizes = [.. outSizes.EnumerateArray().Select(size => size.GetInt32())];
        }

        if (written.TryGetProperty("outputPadding", out var outputPadding))
        {
            OutputPadding = [.. outputPadding.EnumerateArray().Select(places => places.GetInt32())];
        }

        if (written.TryGetProperty("sizes", out var sizes))
        {
            Sizes = [.. sizes.EnumerateArray().Select(size => size.GetInt32())];
            Stride = Wholes(written, "stride");
            Dilation = Wholes(written, "dilation");
        }

        if (written.TryGetProperty("padding", out var padding))
        {
            switch (padding.ValueKind)
            {
                case JsonValueKind.Number:
                    Padding = padding.GetInt32();
                    break;
                case JsonValueKind.String:
                    PaddingMode = padding.GetString() == "same" ? PaddingMode.Same : PaddingMode.Causal;
                    break;
                default:
                    Border = [.. padding.EnumerateArray().Select(places => places.GetInt32())];
                    break;
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

    /// <summary>
    /// What it is: <c>convolution</c>, <c>transposed</c>, <c>max</c>, <c>average</c>, <c>globalAverage</c>, <c>globalMax</c>,
    /// <c>adaptiveAverage</c>, <c>adaptiveMax</c> or <c>pad</c>.
    /// </summary>
    public string Kind { get; }

    /// <summary>How many axes the window walks: one, two or three.</summary>
    public int Rank { get; }

    public int Count { get; }

    public int[] Extents { get; }

    public int Channels { get; }

    public int OutChannels { get; }

    /// <summary>How many groups the channels are split into; one, where the case says nothing of it.</summary>
    public int Groups { get; }

    public int[] Sizes { get; } = [];

    /// <summary>How far the window moves along each axis.</summary>
    public int[] Stride { get; } = [];

    /// <summary>How far apart the window's cells are along each axis.</summary>
    public int[] Dilation { get; } = [];

    /// <summary>The places of nothing a transposed convolution adds at the end of each axis.</summary>
    public int[] OutputPadding { get; } = [];

    /// <summary>What a pooling to a size other than one place pools each axis to.</summary>
    public int[] OutSizes { get; } = [];

    public int Padding { get; }

    public PaddingMode PaddingMode { get; }

    /// <summary>The places of nothing at each end of each axis, where the case states them apart; nothing otherwise.</summary>
    public int[] Border { get; } = [];

    /// <summary>Whether the window also stands at a last place whose patch overhangs the end.</summary>
    public bool RoundsUp { get; }

    public bool CountsPadding { get; }

    /// <summary>What a layer that only pads pads with.</summary>
    public float Value { get; }

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

    /// <summary>The names of the cases of one kind whose channels are all in one group: what a layer that does not group reads.</summary>
    public static TheoryData<string> NamedUngrouped(string kind, params int[] ranks) =>
        [.. All.Where(reference => reference.Kind == kind && reference.Groups == 1 && (ranks.Length == 0 || ranks.Contains(reference.Rank)))
            .Select(reference => reference.Name)];

    /// <summary>The case of a name.</summary>
    public static SpatialReference Of(string name) => All.Single(reference => reference.Name == name);

    /// <summary>The images, volumes or series the case handed PyTorch, channels last.</summary>
    public Tensor Images() => Tensor.From(new Shape([Count, .. Extents, Channels]), Input);

    /// <summary>The gradient of the loss at the layer's output: what the case multiplied it by before adding it up.</summary>
    public Tensor Gradient() => Tensor.From(OutShape, GradOutput);

    public Window1D Along() => new(Sizes[0])
    {
        Stride = Stride[0],
        Dilation = Dilation[0],
        RoundsUp = RoundsUp,
        Padding = Border.Length > 0 ? 0 : Padding,
        PaddingMode = Border.Length > 0 ? PaddingMode.Stated : PaddingMode,
        Border = Border.Length > 0 ? new Sides(Border[0], Border[1]) : null,
    };

    public Window Over() => new(Sizes[0], Sizes[1])
    {
        Stride = new Steps(Stride[0], Stride[1]),
        Dilation = new Steps(Dilation[0], Dilation[1]),
        RoundsUp = RoundsUp,
        Padding = Border.Length > 0 ? 0 : Padding,
        PaddingMode = Border.Length > 0 ? PaddingMode.Stated : PaddingMode,
        Border = Border.Length > 0 ? new Margins(new Sides(Border[0], Border[1]), new Sides(Border[2], Border[3])) : null,
    };

    public Window3D Through() => new(Sizes[0], Sizes[1], Sizes[2])
    {
        Stride = new Steps3D(Stride[0], Stride[1], Stride[2]),
        Dilation = new Steps3D(Dilation[0], Dilation[1], Dilation[2]),
        RoundsUp = RoundsUp,
        Padding = Border.Length > 0 ? 0 : Padding,
        PaddingMode = Border.Length > 0 ? PaddingMode.Stated : PaddingMode,
        Border = Border.Length > 0
            ? new Margins3D(new Sides(Border[0], Border[1]), new Sides(Border[2], Border[3]), new Sides(Border[4], Border[5]))
            : null,
    };

    /// <summary>The kernel and bias PyTorch had, as the convolution of the case's rank.</summary>
    public Convolution Convolution()
    {
        var kernel = Tensor.From(new Shape(Kernel.Length / OutChannels, OutChannels), Kernel);
        var bias = Tensor.From(new Shape(OutChannels), Bias);

        var channels = new Channels { In = Channels, Out = OutChannels, Groups = Groups };

        return Rank switch
        {
            1 => new Conv1D(kernel, bias, Along(), channels),
            2 => new Conv2D(kernel, bias, Over(), channels),
            _ => new Conv3D(kernel, bias, Through(), channels),
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
        ("adaptiveAverage", 1) => new AdaptiveAvgPool1D(Pooled()),
        ("adaptiveAverage", 2) => new AdaptiveAvgPool2D(Pooled()),
        ("adaptiveAverage", 3) => new AdaptiveAvgPool3D(Pooled()),
        ("adaptiveMax", 1) => new AdaptiveMaxPool1D(Pooled()),
        ("adaptiveMax", 2) => new AdaptiveMaxPool2D(Pooled()),
        ("adaptiveMax", 3) => new AdaptiveMaxPool3D(Pooled()),
        _ => throw new InvalidOperationException($"No layer pools as '{Kind}' along {Rank} axes."),
    };

    /// <summary>The kernel and bias PyTorch had, as the transposed convolution of the case's rank.</summary>
    public Convolution Transposed()
    {
        var kernel = Tensor.From(new Shape(Channels, Kernel.Length / Channels), Kernel);
        var bias = Tensor.From(new Shape(OutChannels), Bias);
        var channels = new Channels { In = Channels, Out = OutChannels, Groups = Groups };

        return Rank switch
        {
            1 => new Conv1DTranspose(kernel, bias, Along(), channels),
            2 => new Conv2DTranspose(kernel, bias, Over(), channels),
            _ => new Conv3DTranspose(kernel, bias, Through(), channels),
        };
    }

    /// <summary>
    /// The same transposed convolution, built as PyTorch builds one — with the places it adds at the end of each axis — and
    /// then handed PyTorch's own numbers through the door numbers trained elsewhere go in by.
    /// </summary>
    public Convolution TransposedGrown()
    {
        var channels = new Channels { In = Channels, Out = OutChannels, Groups = Groups };
        var draws = new RandomStream(1).Draw("reference", 0, 0);

        Convolution layer = Rank switch
        {
            1 => new Conv1DTranspose(channels, Along(), draws, OutputPadding[0]),
            2 => new Conv2DTranspose(channels, Over(), draws, new Steps(OutputPadding[0], OutputPadding[1])),
            _ => new Conv3DTranspose(channels, Through(), draws, new Steps3D(OutputPadding[0], OutputPadding[1], OutputPadding[2])),
        };

        new LayerStack(layer).Load([
            new SlotEntry("0.weight", Tensor.From(new Shape(Channels, Kernel.Length / Channels), Kernel), "the reference"),
            new SlotEntry("0.bias", Tensor.From(new Shape(OutChannels), Bias), "the reference"),
        ]);

        return layer;
    }

    /// <summary>What a pooling to a size other than one place pools its axes to.</summary>
    public Shape Pooled() => new(OutSizes);

    /// <summary>The layer that only pads, of the case's rank, with the places and the value the case holds.</summary>
    public Layer Pads() => Rank switch
    {
        1 => new Pad1D(new Sides(Border[0], Border[1]), Value),
        2 => new Pad2D(new Margins(new Sides(Border[0], Border[1]), new Sides(Border[2], Border[3])), Value),
        _ => new Pad3D(
            new Margins3D(new Sides(Border[0], Border[1]), new Sides(Border[2], Border[3]), new Sides(Border[4], Border[5])),
            Value),
    };

    /// <summary>The sum of the layer's output times the gradient the case multiplied it by: the loss PyTorch took the gradient of.</summary>
    public Tensor LossOver(ITensorBackend backend, Tensor output) =>
        backend.Scale(backend.Mean(backend.Multiply(output, Gradient())), backend.Fill(new Shape(), output.Shape.Count));

    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{Name}: {Count}x{string.Join('x', Extents)}x{Channels}");

    private static List<SpatialReference> Read()
    {
        var path = Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "spatial-pytorch.json");

        using var document = JsonDocument.Parse(File.ReadAllText(path));

        return [.. document.RootElement.GetProperty("cases").EnumerateArray().Select(written => new SpatialReference(written))];
    }

    private static int[] Wholes(JsonElement written, string name) => [.. written.GetProperty(name).EnumerateArray().Select(length => length.GetInt32())];

    private static float[] Values(JsonElement written, string name) =>
        [.. written.GetProperty(name).EnumerateArray().Select(value => (float)value.GetDouble())];
}
