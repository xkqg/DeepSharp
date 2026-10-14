// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A convolution that gives back the places an ordinary one takes away: Keras's <c>Conv1DTranspose</c>,
/// <c>Conv2DTranspose</c> and <c>Conv3DTranspose</c>, PyTorch's <c>ConvTranspose</c>, ONNX's <c>ConvTranspose</c>.
/// </summary>
/// <remarks>
/// It is the ordinary convolution's way back, run forwards. Each place of the example is multiplied by the kernel into a
/// patch of the window's own size, and every patch is put back where the window would have taken it from, the values that
/// land on the same place added together — which is the folding the engines already do, because it is how the gradient of
/// an unfolding goes back. So nothing is added to the tensor seam for it, and its own gradient is an unfolding.
/// <para>
/// The example comes out as many places as the window would have been slid over to make it: along each axis,
/// <c>(places − 1) × stride + dilation × (size − 1) + 1</c>, less the border stated at each end, and so many more as
/// <see cref="TransposedConvolution.OutputPadding"/> says — the one thing that is not worked out, because a window of a given
/// stride takes several sizes down to the same one and only the layer can say which of them it came from. Those places are
/// not empty ones added afterwards: the folding reaches into them, so what the output padding does is let that much LESS be
/// dropped at the end, and only where it asks for more than the border had is anything added. Its border is stated and never
/// worked out: a window padded as 'same' pads by what the example reaching it is, which is the one thing a layer that makes
/// its own example cannot read.
/// </para>
/// </remarks>
public abstract class TransposedConvolution : Convolution
{
    private readonly int _rank;

    private protected TransposedConvolution(Starting starting, Beginning beginning, int rank)
        : base(starting, beginning) => _rank = rank;

    private protected TransposedConvolution(Starting starting, Tensor kernel, Tensor? bias)
        : base(starting, kernel, bias) => _rank = starting.Walk.Rank;

    /// <summary>How many places of nothing are added at the end of each axis, beyond what the window's own walk gives back.</summary>
    /// <remarks>Fewer than the larger of the stride and the dilation of that axis: beyond that the window itself gives the place.</remarks>
    public abstract IReadOnlyList<int> OutputPadding { get; }

    /// <summary>
    /// A row for every channel in, holding the window's places by the channels out of one group: the shape a transposed
    /// convolution's kernel keeps, and what it starts from.
    /// </summary>
    /// <param name="walk">How its window walks the example it makes.</param>
    /// <param name="channels">The channels it was told of; a transposed convolution is always told them.</param>
    /// <returns>What a convolution of this family starts from.</returns>
    private protected static Starting GivingBack(ISpatialWalk walk, Channels channels) =>
        new(walk, channels, new Shape(channels.In, walk.Volume * channels.Out / channels.Groups));

    /// <inheritdoc />
    private protected sealed override string Named => $"A transposed convolution of {InChannels} channels";

    /// <summary>The walk that adds the places at the end of each axis, where any are added; nothing where none are.</summary>
    private protected abstract ISpatialWalk? Grows { get; }

    /// <inheritdoc />
    private protected sealed override void RequireInput(Shape batch) => batch.RequireSpatial(_rank, InChannels, Named);

    /// <inheritdoc />
    private protected sealed override Tensor Through(ITensorBackend backend, Tensor input)
    {
        var extents = Walk.ExtentsOf(input.Shape);
        var canvas = Canvas(extents);
        var patches = backend.MatMul(
            backend.Reshape(input, new Shape(input.Shape.Count / Channels.In, Channels.In)),
            Channels.Groups == 1 ? Weight.Value : Spread(backend));
        var made = Walk.Fold(backend, patches, canvas);
        var grown = Grows is { } growth ? growth.Padded(backend, made, 0f) : made;

        return Bias is null
            ? grown
            : backend.Reshape(
                backend.AddRow(backend.Reshape(grown, new Shape(grown.Shape.Count / Channels.Out, Channels.Out)), Bias.Value),
                grown.Shape);
    }

    /// <summary>How many places a window gives back along one axis, of an example so many places long.</summary>
    /// <param name="places">How many places the axis reaching the layer has.</param>
    /// <param name="stride">How far the window moves along it.</param>
    /// <param name="reach">How many places the window spans, the gaps of a dilated one included.</param>
    /// <param name="dropped">What the folding drops at each end of it.</param>
    /// <returns>The places, which is nought or less where the border takes more than the window gives.</returns>
    /// <remarks>The one rule for the size a transposed convolution makes, read both by the layer and by a description that holds one.</remarks>
    public static int GivesBack(int places, int stride, int reach, Sides dropped) =>
        ((places - 1) * stride) + reach - dropped.Total;

    /// <summary>How long each axis of the example is before the places at its end are added: what the window's walk gives back.</summary>
    /// <param name="extents">How long each axis of an example reaching the layer is.</param>
    /// <returns>The lengths, one for each axis the window walks.</returns>
    /// <exception cref="ArgumentException">The window gives back no place along an axis.</exception>
    private protected int[] Canvas(int[] extents)
    {
        var canvas = new int[_rank];

        for (var axis = 0; axis < _rank; axis++)
        {
            canvas[axis] = GivesBack(extents[axis], Stride(axis), (Dilation(axis) * (Size(axis) - 1)) + 1, Sides(axis));

            if (canvas[axis] < 1)
            {
                throw new ArgumentException(
                    $"{Named} gives back {canvas[axis]} places along an axis of {extents[axis]}, and an example holds one place at least: its border takes more than its window gives.",
                    "input");
            }
        }

        return canvas;
    }

    /// <summary>How far the window moves along an axis.</summary>
    private protected abstract int Stride(int axis);

    /// <summary>How far apart the window's cells are along an axis.</summary>
    private protected abstract int Dilation(int axis);

    /// <summary>How many places of the axis the window covers.</summary>
    private protected abstract int Size(int axis);

    /// <summary>
    /// The places dropped at each end of an axis once the folding has put the patches back: the border stated before the
    /// example, and after it the border less the places the output padding asks to keep, never fewer than none.
    /// </summary>
    private protected abstract Sides Sides(int axis);

    /// <summary>The border a window states at each end of an axis, as the layer was given it.</summary>
    /// <param name="sides">The border given.</param>
    /// <param name="outputPadding">How many places the output padding asks for at the end.</param>
    /// <returns>What the folding drops, and what is added after it.</returns>
    /// <remarks>
    /// The folding reaches as far as the window does, so a place the output padding asks for is a place already made: it is
    /// kept by dropping less, and only where it asks for more places than the border had is anything added at the end.
    /// </remarks>
    public static (Sides Dropped, int Added) Settling(Sides sides, int outputPadding) =>
        (new Sides(sides.Before, Math.Max(0, sides.After - outputPadding)), Math.Max(0, outputPadding - sides.After));

    /// <summary>Refuses an output padding that reaches as far as the window's own walk does, and a border that is worked out.</summary>
    /// <param name="worked">Whether the window works its border out from the example rather than stating it.</param>
    /// <exception cref="ArgumentException">The output padding is as large as the stride or the dilation, or the border is worked out.</exception>
    private protected void RequireGrowth(bool worked)
    {
        if (worked)
        {
            throw new ArgumentException(
                $"{Named} makes the example its window is walked over, so it cannot read a border worked out from one: its border is stated at each end.",
                "window");
        }

        for (var axis = 0; axis < _rank; axis++)
        {
            var reach = Math.Max(Stride(axis), Dilation(axis));

            if (OutputPadding[axis] < 0 || OutputPadding[axis] >= reach)
            {
                throw new ArgumentException(
                    $"{Named} adds {OutputPadding[axis]} places at the end of an axis it strides {Stride(axis)} and dilates {Dilation(axis)}, and adds fewer than {reach}: beyond that the window gives the place itself.",
                    "outputPadding");
            }
        }
    }

    // The kernel every channel out needs, made from the one the layer holds: each group's columns spread into the places its
    // channels out stand at, and nought wherever a channel in and a channel out are not in the same group. Both depend on the
    // shapes alone, so what the kernel learns is what a grouped transposed convolution learns.
    private Tensor Spread(ITensorBackend backend)
    {
        var inGroup = Channels.In / Channels.Groups;
        var outGroup = Channels.Out / Channels.Groups;
        var held = Walk.Volume * outGroup;
        var columns = Walk.Volume * Channels.Out;
        var spreading = new float[held * columns];
        var mask = new float[Channels.In * columns];

        for (var place = 0; place < Walk.Volume; place++)
        {
            for (var made = 0; made < outGroup; made++)
            {
                for (var group = 0; group < Channels.Groups; group++)
                {
                    spreading[(((place * outGroup) + made) * columns) + (place * Channels.Out) + (group * outGroup) + made] = 1f;
                }
            }
        }

        for (var channel = 0; channel < Channels.In; channel++)
        {
            for (var place = 0; place < Walk.Volume; place++)
            {
                for (var made = channel / inGroup * outGroup; made < ((channel / inGroup) + 1) * outGroup; made++)
                {
                    mask[(channel * columns) + (place * Channels.Out) + made] = 1f;
                }
            }
        }

        return backend.Multiply(
            backend.MatMul(Weight.Value, Tensor.From(new Shape(held, columns), spreading)),
            Tensor.From(new Shape(Channels.In, columns), mask));
    }
}

/// <summary>A convolution along a series that gives back the steps an ordinary one takes away.</summary>
public sealed class Conv1DTranspose : TransposedConvolution, ISaved<Conv1DTranspose>
{
    /// <summary>A transposed convolution that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks the series it makes.</param>
    /// <param name="beginning">The draws its start is taken from, and how its kernel and its bias start — or that it has no bias.</param>
    /// <param name="outputPadding">How many steps of nothing to add at the end of the series; none, unless said.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    /// <exception cref="ArgumentNullException">The beginning holds no draws.</exception>
    public Conv1DTranspose(Channels channels, Window1D window, Beginning beginning, int outputPadding = 0)
        : base(GivingBack(new LineWalk(Folding(window, outputPadding)), channels), beginning, 1)
    {
        Window = window;
        Grew = outputPadding;
        Grows = Growth(Settling(Stated, outputPadding).Added);
        RequireGrowth(window.PaddingMode != PaddingMode.Stated);
    }

    /// <summary>A transposed convolution that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <param name="outputPadding">How many places of nothing to add at the end of each axis; none, unless said.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Conv1DTranspose(Channels channels, Window1D window, Draws draws, int outputPadding = 0)
        : this(channels, window, new Beginning(draws), outputPadding)
    {
    }

    /// <summary>A transposed convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">A row for every channel in, holding the window's steps by the channels out of one group.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks the series it makes.</param>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <exception cref="ArgumentException">The kernel is not the shape the window and the channels ask for, or the window works its border out.</exception>
    public Conv1DTranspose(Tensor kernel, Tensor? bias, Window1D window, Channels channels)
        : base(GivingBack(new LineWalk(Folding(window, 0)), channels), kernel, bias)
    {
        Window = window;
        Grows = null;
        RequireGrowth(window.PaddingMode != PaddingMode.Stated);
    }

    /// <summary>The patch, and how it walks the series it makes.</summary>
    public Window1D Window { get; }

    /// <inheritdoc />
    public override IReadOnlyList<int> OutputPadding => [Grew];

    /// <inheritdoc />
    public static string Name => "conv1dtranspose";

    private int Grew { get; }

    /// <inheritdoc />
    private protected override ISpatialWalk? Grows { get; }

    /// <inheritdoc />
    public static Conv1DTranspose Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(
            ChannelsIn(settings, rebuilding),
            rebuilding.Window1DIn(settings),
            BeginningIn(settings, rebuilding),
            rebuilding.Whole(settings, "outputPadding", 0));
    }

    /// <inheritdoc />
    public int FileVersion => SavedAs(Window.FileVersion);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteChannels(writer);

        if (Grew != 0)
        {
            writer.WriteNumber("outputPadding", Grew);
        }

        writer.WriteWindow(Window);
    }

    /// <inheritdoc />
    private protected override int Stride(int axis) => Window.Stride;

    /// <inheritdoc />
    private protected override int Dilation(int axis) => Window.Dilation;

    /// <inheritdoc />
    private protected override int Size(int axis) => Window.Length;

    /// <inheritdoc />
    private protected override Sides Sides(int axis) => Settling(Stated, Grew).Dropped;

    // The border the window was given, whichever way it says it.
    private Sides Stated => Window.Border ?? new Sides(Window.Padding, Window.Padding);

    // The window the folding goes over: the same walk, dropping at the end only what the output padding does not keep.
    private static Window1D Folding(Window1D window, int outputPadding)
    {
        var stated = window.Border ?? new Sides(window.Padding, window.Padding);

        return window with { Padding = 0, PaddingMode = PaddingMode.Stated, Border = Settling(stated, outputPadding).Dropped };
    }

    private static ISpatialWalk? Growth(int added) =>
        added == 0 ? null : new LineWalk(new Window1D(1) { Border = new Sides(0, added) });
}

/// <summary>A convolution over an image that gives back the rows and columns an ordinary one takes away.</summary>
public sealed class Conv2DTranspose : TransposedConvolution, ISaved<Conv2DTranspose>
{
    /// <summary>A transposed convolution that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks the image it makes.</param>
    /// <param name="beginning">The draws its start is taken from, and how its kernel and its bias start — or that it has no bias.</param>
    /// <param name="outputPadding">How many rows and columns of nothing to add at the end of each axis; none, unless said.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    /// <exception cref="ArgumentNullException">The beginning holds no draws.</exception>
    public Conv2DTranspose(Channels channels, Window window, Beginning beginning, Steps outputPadding = default)
        : base(GivingBack(new PlaneWalk(Folding(window, outputPadding)), channels), beginning, 2)
    {
        Window = window;
        Grew = outputPadding;
        Grows = Growth();
        RequireGrowth(window.PaddingMode != PaddingMode.Stated);
    }

    /// <summary>A transposed convolution that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <param name="outputPadding">How many places of nothing to add at the end of each axis; none, unless said.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Conv2DTranspose(Channels channels, Window window, Draws draws, Steps outputPadding = default)
        : this(channels, window, new Beginning(draws), outputPadding)
    {
    }

    /// <summary>A transposed convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">A row for every channel in, holding the window's places by the channels out of one group.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks the image it makes.</param>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <exception cref="ArgumentException">The kernel is not the shape the window and the channels ask for, or the window works its border out.</exception>
    public Conv2DTranspose(Tensor kernel, Tensor? bias, Window window, Channels channels)
        : base(GivingBack(new PlaneWalk(Folding(window, default)), channels), kernel, bias)
    {
        Window = window;
        Grows = null;
        RequireGrowth(window.PaddingMode != PaddingMode.Stated);
    }

    /// <summary>The patch, and how it walks the image it makes.</summary>
    public Window Window { get; }

    /// <inheritdoc />
    public override IReadOnlyList<int> OutputPadding => [Grew.Down, Grew.Across];

    /// <inheritdoc />
    public static string Name => "conv2dtranspose";

    private Steps Grew { get; }

    /// <inheritdoc />
    private protected override ISpatialWalk? Grows { get; }

    /// <inheritdoc />
    public static Conv2DTranspose Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var grown = rebuilding.Wholes(settings, "outputPadding", [0, 0]);

        return new(
            ChannelsIn(settings, rebuilding),
            rebuilding.Window2DIn(settings),
            BeginningIn(settings, rebuilding),
            new Steps(grown[0], grown[1]));
    }

    /// <inheritdoc />
    public int FileVersion => SavedAs(Window.FileVersion);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteChannels(writer);

        if (Grew != default)
        {
            writer.WriteStartArray("outputPadding");
            writer.WriteNumberValue(Grew.Down);
            writer.WriteNumberValue(Grew.Across);
            writer.WriteEndArray();
        }

        writer.WriteWindow(Window);
    }

    /// <inheritdoc />
    private protected override int Stride(int axis) => axis == 0 ? Window.Stride.Down : Window.Stride.Across;

    /// <inheritdoc />
    private protected override int Dilation(int axis) => axis == 0 ? Window.Dilation.Down : Window.Dilation.Across;

    /// <inheritdoc />
    private protected override int Size(int axis) => axis == 0 ? Window.Height : Window.Width;

    /// <inheritdoc />
    private protected override Sides Sides(int axis) => Settling(Stated(axis), axis == 0 ? Grew.Down : Grew.Across).Dropped;

    // The border the window was given along one axis, whichever way it says it.
    private Sides Stated(int axis) =>
        Window.Border is { } sides ? (axis == 0 ? sides.Down : sides.Across) : new Sides(Window.Padding, Window.Padding);

    // The window the folding goes over: the same walk, dropping at each end only what the output padding does not keep.
    private static Window Folding(Window window, Steps outputPadding)
    {
        var down = window.Border is { } sides ? sides.Down : new Sides(window.Padding, window.Padding);
        var across = window.Border is { } also ? also.Across : new Sides(window.Padding, window.Padding);

        return window with
        {
            Padding = 0,
            PaddingMode = PaddingMode.Stated,
            Border = new Margins(Settling(down, outputPadding.Down).Dropped, Settling(across, outputPadding.Across).Dropped),
        };
    }

    private ISpatialWalk? Growth()
    {
        var added = new Steps(Settling(Stated(0), Grew.Down).Added, Settling(Stated(1), Grew.Across).Added);

        return added == default
            ? null
            : new PlaneWalk(new Window(1, 1) { Border = new Margins(new Sides(0, added.Down), new Sides(0, added.Across)) });
    }
}

/// <summary>A convolution through a volume that gives back the places an ordinary one takes away.</summary>
public sealed class Conv3DTranspose : TransposedConvolution, ISaved<Conv3DTranspose>
{
    /// <summary>A transposed convolution that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The block, and how it walks the volume it makes.</param>
    /// <param name="beginning">The draws its start is taken from, and how its kernel and its bias start — or that it has no bias.</param>
    /// <param name="outputPadding">How many places of nothing to add at the end of each axis; none, unless said.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    /// <exception cref="ArgumentNullException">The beginning holds no draws.</exception>
    public Conv3DTranspose(Channels channels, Window3D window, Beginning beginning, Steps3D outputPadding = default)
        : base(GivingBack(new VolumeWalk(Folding(window, outputPadding)), channels), beginning, 3)
    {
        Window = window;
        Grew = outputPadding;
        Grows = Growth();
        RequireGrowth(window.PaddingMode != PaddingMode.Stated);
    }

    /// <summary>A transposed convolution that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <param name="outputPadding">How many places of nothing to add at the end of each axis; none, unless said.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Conv3DTranspose(Channels channels, Window3D window, Draws draws, Steps3D outputPadding = default)
        : this(channels, window, new Beginning(draws), outputPadding)
    {
    }

    /// <summary>A transposed convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">A row for every channel in, holding the window's places by the channels out of one group.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The block, and how it walks the volume it makes.</param>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <exception cref="ArgumentException">The kernel is not the shape the window and the channels ask for, or the window works its border out.</exception>
    public Conv3DTranspose(Tensor kernel, Tensor? bias, Window3D window, Channels channels)
        : base(GivingBack(new VolumeWalk(Folding(window, default)), channels), kernel, bias)
    {
        Window = window;
        Grows = null;
        RequireGrowth(window.PaddingMode != PaddingMode.Stated);
    }

    /// <summary>The block, and how it walks the volume it makes.</summary>
    public Window3D Window { get; }

    /// <inheritdoc />
    public override IReadOnlyList<int> OutputPadding => [Grew.Deep, Grew.Down, Grew.Across];

    /// <inheritdoc />
    public static string Name => "conv3dtranspose";

    private Steps3D Grew { get; }

    /// <inheritdoc />
    private protected override ISpatialWalk? Grows { get; }

    /// <inheritdoc />
    public static Conv3DTranspose Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var grown = rebuilding.Wholes(settings, "outputPadding", [0, 0, 0]);

        return new(
            ChannelsIn(settings, rebuilding),
            rebuilding.Window3DIn(settings),
            BeginningIn(settings, rebuilding),
            new Steps3D(grown[0], grown[1], grown[2]));
    }

    /// <inheritdoc />
    public int FileVersion => SavedAs(Window.FileVersion);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteChannels(writer);

        if (Grew != default)
        {
            writer.WriteStartArray("outputPadding");
            writer.WriteNumberValue(Grew.Deep);
            writer.WriteNumberValue(Grew.Down);
            writer.WriteNumberValue(Grew.Across);
            writer.WriteEndArray();
        }

        writer.WriteWindow(Window);
    }

    /// <inheritdoc />
    private protected override int Stride(int axis) => axis switch { 0 => Window.Stride.Deep, 1 => Window.Stride.Down, _ => Window.Stride.Across };

    /// <inheritdoc />
    private protected override int Dilation(int axis) => axis switch { 0 => Window.Dilation.Deep, 1 => Window.Dilation.Down, _ => Window.Dilation.Across };

    /// <inheritdoc />
    private protected override int Size(int axis) => axis switch { 0 => Window.Depth, 1 => Window.Height, _ => Window.Width };

    /// <inheritdoc />
    private protected override Sides Sides(int axis) =>
        Settling(Stated(axis), axis switch { 0 => Grew.Deep, 1 => Grew.Down, _ => Grew.Across }).Dropped;

    // The border the window was given along one axis, whichever way it says it.
    private Sides Stated(int axis) =>
        Window.Border is { } sides
            ? axis switch { 0 => sides.Deep, 1 => sides.Down, _ => sides.Across }
            : new Sides(Window.Padding, Window.Padding);

    // The window the folding goes over: the same walk, dropping at each end only what the output padding does not keep.
    private static Window3D Folding(Window3D window, Steps3D outputPadding)
    {
        var even = new Sides(window.Padding, window.Padding);
        var stated = window.Border ?? new Margins3D(even, even, even);

        return window with
        {
            Padding = 0,
            PaddingMode = PaddingMode.Stated,
            Border = new Margins3D(
                Settling(stated.Deep, outputPadding.Deep).Dropped,
                Settling(stated.Down, outputPadding.Down).Dropped,
                Settling(stated.Across, outputPadding.Across).Dropped),
        };
    }

    private ISpatialWalk? Growth()
    {
        var added = new Steps3D(
            Settling(Stated(0), Grew.Deep).Added,
            Settling(Stated(1), Grew.Down).Added,
            Settling(Stated(2), Grew.Across).Added);

        return added == default
            ? null
            : new VolumeWalk(new Window3D(1, 1, 1)
            {
                Border = new Margins3D(new Sides(0, added.Deep), new Sides(0, added.Down), new Sides(0, added.Across)),
            });
    }
}
