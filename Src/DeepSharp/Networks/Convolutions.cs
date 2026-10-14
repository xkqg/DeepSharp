// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A convolution along one, two or three axes: a window walked over each example, every place it stands given a value for
/// each channel the layer makes — the patch it covers, times a kernel, plus a bias.
/// </summary>
/// <remarks>
/// What the three share is written once, here: the kernel — as many rows as the window holds values, channels in included,
/// by as many columns as channels out, starting as PyTorch's does, within one over the root of the values a window holds —
/// the bias, and the arithmetic. Each says only the window it walks and how that is written to a network file. Examples are
/// laid out with their channels last, as Keras lays them out, so a batch is the patches of every example one after another
/// and nothing has to be turned round; PyTorch keeps the same numbers channels first.
/// </remarks>
public abstract class Convolution : Layer
{
    private readonly ISpatialWalk _walk;

    /// <summary>A convolution that starts as PyTorch's does, unless the beginning says otherwise.</summary>
    private protected Convolution(Starting starting, Beginning beginning)
    {
        var begun = beginning.Required();
        var walk = starting.Walk;

        _walk = walk;
        Channels = RequireChannels(starting.Channels!.Value, walk);
        KernelShape = starting.Kernel!.Value;

        var fans = new Fans(walk.Volume * Channels.In / Channels.Groups, walk.Volume * Channels.Out);

        Weight = AddParameter("weight", begun.WeightsOf(KernelShape, fans));
        Bias = begun.BiasOf(new Shape(Channels.Out), fans) is { } bias ? AddParameter("bias", bias) : null;
    }

    /// <summary>A convolution that starts at the given kernel and bias; with no bias where none is given.</summary>
    /// <exception cref="ArgumentException">The kernel is not the shape the window and the channels ask for, or the bias is not as long as its channels out.</exception>
    private protected Convolution(Starting starting, Tensor kernel, Tensor? bias)
    {
        ArgumentNullException.ThrowIfNull(kernel);

        var walk = starting.Walk;

        _walk = walk;

        if (starting.Channels is { } given)
        {
            Channels = RequireChannels(given, walk);
            KernelShape = starting.Kernel!.Value;

            if (kernel.Shape != KernelShape || (bias is not null && bias.Shape != new Shape(Channels.Out)))
            {
                throw new ArgumentException(
                    $"A convolution through a {walk.Describe()} of {Channels} takes a {KernelShape} kernel and a bias as long as "
                    + $"its channels out; these are {kernel.Shape} and {bias?.Shape.ToString() ?? "no bias"}.",
                    nameof(kernel));
            }
        }
        else
        {
            // A kernel handed on its own says the channels it holds, every one of them in the same group.
            if (kernel.Shape.Rank != 2 || kernel.Shape[0] < 1 || kernel.Shape[0] % walk.Volume != 0 || (bias is not null && bias.Shape != new Shape(kernel.Shape[1])))
            {
                throw new ArgumentException(
                    $"A convolution through a {walk.Describe()} takes a kernel of the window's {walk.Volume} places times its channels in by its channels out, "
                    + $"and a bias as long as its channels out; these are {kernel.Shape} and {bias?.Shape.ToString() ?? "no bias"}.",
                    nameof(kernel));
            }

            Channels = RequireChannels(new Channels { In = kernel.Shape[0] / walk.Volume, Out = kernel.Shape[1] }, walk);
            KernelShape = kernel.Shape;
        }

        Weight = AddParameter("weight", kernel);
        Bias = bias is null ? null : AddParameter("bias", bias);
    }

    /// <summary>The kernel: the window's values and the channels in of one group, by the channels out.</summary>
    public Parameter Weight { get; }

    /// <summary>The bias: one value for each channel out; nothing for a convolution built without one.</summary>
    public Parameter? Bias { get; }

    /// <summary>How many channels each place of an example holds, how many the layer makes, and how many groups they are split into.</summary>
    public Channels Channels { get; }

    /// <summary>How many channels each place of an example holds.</summary>
    public int InChannels => Channels.In;

    /// <summary>How many channels the layer makes.</summary>
    public int OutChannels => Channels.Out;

    /// <summary>How many groups the channels are split into; one, where every channel in reaches every channel out.</summary>
    public int Groups => Channels.Groups;

    /// <summary>The shape this convolution's kernel keeps, as the family it belongs to lays one out.</summary>
    private protected Shape KernelShape { get; }

    /// <summary>
    /// The window's values and the channels in of one group, by the channels out: the shape an ordinary convolution's kernel
    /// keeps, and what it starts from.
    /// </summary>
    /// <param name="walk">How its window walks.</param>
    /// <param name="channels">The channels it was told of; nothing where they are to be read off the kernel.</param>
    /// <returns>What a convolution of this family starts from.</returns>
    private protected static Starting Taking(ISpatialWalk walk, Channels? channels) =>
        new(walk, channels, channels is { } given ? new Shape(walk.Volume * given.In / given.Groups, given.Out) : null);

    /// <summary>How the window walks, for what a convolution of another kind needs of it.</summary>
    private protected ISpatialWalk Walk => _walk;

    /// <summary>What this layer is, as a refusal says it.</summary>
    private protected virtual string Named => $"A convolution of {InChannels} channels";

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of what the window walks, holds another number of channels, or is smaller than the window with its border.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        RequireInput(input.Shape);

        return Through(pass.Backend, input);
    }

    /// <summary>Refuses a batch this layer cannot take: not what the window walks, other channels, or smaller than the window.</summary>
    /// <param name="batch">The shape of the batch.</param>
    /// <exception cref="ArgumentException">The batch is one the layer cannot take.</exception>
    /// <remarks>A convolution walks its window over what it is handed; one that gives the places back walks it over what it makes.</remarks>
    private protected virtual void RequireInput(Shape batch) => _walk.RequireBatch(batch, InChannels, Named);

    /// <summary>What the layer makes of a batch its window walks: the arithmetic, the bias and the shape it comes out as.</summary>
    /// <param name="backend">The engine the arithmetic runs on.</param>
    /// <param name="input">The batch, already held to what the window walks.</param>
    /// <returns>What the layer makes of it.</returns>
    /// <remarks>
    /// Three ways, by the groups the channels are in: every channel in one group is the patches times the kernel; a group for
    /// every channel — a depthwise convolution — is each channel's own kernel spread over every place and multiplied in, which
    /// costs a fraction of what the same work costs as a kernel of mostly noughts; and groups between the two spread the
    /// kernel the layer holds into the places the groups give it, by a product that depends on the shapes and not the values,
    /// so what the kernel learns is what a grouped convolution learns.
    /// <para>
    /// A value that is not a number reaches every channel a product reads, because nought times such a value is not a number
    /// either: through a convolution of some groups, whose kernel is spread into their places, it therefore reaches every
    /// channel out and not its own group's alone, and through a depthwise convolution, where a channel is multiplied by its
    /// own kernel and no other is read, it reaches that channel alone. Every number a network holds is finite, so this is
    /// where the arithmetic stands rather than something a layer decides.
    /// </para>
    /// </remarks>
    private protected virtual Tensor Through(ITensorBackend backend, Tensor input)
    {
        var shape = new Shape([input.Shape[0], .. _walk.PlacesOver(_walk.ExtentsOf(input.Shape)), Channels.Out]);

        var values = Channels.Groups switch
        {
            1 => backend.MatMul(_walk.Unfold(backend, input), Weight.Value),
            var groups when groups == Channels.In => Apart(backend, input),
            _ => backend.MatMul(_walk.Unfold(backend, input), Spread(backend)),
        };

        return backend.Reshape(Bias is null ? values : backend.AddRow(values, Bias.Value), shape);
    }

    // Each channel convolved by a kernel of its own, and nothing added to the seam for it: the patches turned so that a place
    // of the window leads, the kernel spread over every place of every example by one product of a single row, multiplied in,
    // and the window's places added up — which is what a sum down the rows of a matrix is.
    private Tensor Apart(ITensorBackend backend, Tensor input)
    {
        var multiplied = Channels.Out / Channels.In;
        var channels = Channels.In * multiplied;
        var patches = _walk.Unfold(backend, multiplied == 1 ? input : Repeated(backend, input, multiplied));
        var rows = patches.Shape[0];
        var spread = backend.MatMul(
            backend.Reshape(Weight.Value, new Shape(_walk.Volume * channels, 1)),
            backend.Fill(new Shape(1, rows), 1f));
        var taken = backend.Multiply(backend.Transpose(patches), spread);
        var totals = backend.SumRows(backend.Reshape(taken, new Shape(_walk.Volume, channels * rows)));

        return backend.Transpose(backend.Reshape(totals, new Shape(channels, rows)));
    }

    // Every channel of an example so many times over, each copy beside its own: what a depthwise convolution that makes more
    // than one channel of each reads, so the channels it makes stand in the order every framework makes them.
    private static Tensor Repeated(ITensorBackend backend, Tensor input, int times)
    {
        var channels = input.Shape[input.Shape.Rank - 1];
        var one = new float[channels * channels * times];

        for (var channel = 0; channel < channels; channel++)
        {
            for (var copy = 0; copy < times; copy++)
            {
                one[(channel * channels * times) + (channel * times) + copy] = 1f;
            }
        }

        var wider = backend.MatMul(
            backend.Reshape(input, new Shape(input.Shape.Count / channels, channels)),
            Tensor.From(new Shape(channels, channels * times), one));

        return backend.Reshape(wider, new Shape([.. input.Shape.Axes[..^1], channels * times]));
    }

    // The kernel the patches of every channel are multiplied by, made from the one the layer holds: its rows put where each
    // group's channels stand, and nought wherever a channel in and a channel out are not in the same group. Both the placing
    // and the mask are worked out from the shapes alone, so the gradient that reaches the kernel is the grouped one and the
    // noughts between the groups are not numbers anything learns.
    private Tensor Spread(ITensorBackend backend)
    {
        var inGroup = Channels.In / Channels.Groups;
        var outGroup = Channels.Out / Channels.Groups;
        var rows = _walk.Volume * Channels.In;
        var held = _walk.Volume * inGroup;
        var placing = new float[rows * held];
        var mask = new float[rows * Channels.Out];

        for (var place = 0; place < _walk.Volume; place++)
        {
            for (var channel = 0; channel < Channels.In; channel++)
            {
                var row = (place * Channels.In) + channel;

                placing[(row * held) + (place * inGroup) + (channel % inGroup)] = 1f;

                for (var made = channel / inGroup * outGroup; made < ((channel / inGroup) + 1) * outGroup; made++)
                {
                    mask[(row * Channels.Out) + made] = 1f;
                }
            }
        }

        return backend.Multiply(
            backend.MatMul(Tensor.From(new Shape(rows, held), placing), Weight.Value),
            Tensor.From(new Shape(rows, Channels.Out), mask));
    }

    /// <summary>The channels a convolution was written with: the channels in and out, and the groups where it says them.</summary>
    /// <param name="settings">The object the convolution was written as.</param>
    /// <param name="rebuilding">The file's reader.</param>
    /// <returns>The channels.</returns>
    private protected static Channels ChannelsIn(JsonElement settings, Rebuilding rebuilding) => new()
    {
        In = rebuilding.Whole(settings, "inChannels"),
        Out = rebuilding.Whole(settings, "outChannels"),
        Groups = rebuilding.Whole(settings, "groups", 1),
    };

    /// <summary>How a convolution being rebuilt begins: with the draws it is handed, and without a bias where the file says it has none.</summary>
    /// <param name="settings">The object the convolution was written as.</param>
    /// <param name="rebuilding">The file's reader.</param>
    /// <returns>The beginning.</returns>
    private protected static Beginning BeginningIn(JsonElement settings, Rebuilding rebuilding) =>
        new(rebuilding.Draws) { WithoutBias = !rebuilding.Yes(settings, "bias") };

    /// <summary>Writes the channels of this convolution as a file holds them, and that it has no bias where it has none.</summary>
    /// <param name="writer">The writer, inside the object the convolution is written as.</param>
    /// <remarks>The groups are written only when there are some, and the bias only when it is not there, so a convolution that says neither is written as it always was.</remarks>
    private protected void WriteChannels(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("inChannels", InChannels);
        writer.WriteNumber("outChannels", OutChannels);

        if (Groups != 1)
        {
            writer.WriteNumber("groups", Groups);
        }

        if (Bias is null)
        {
            writer.WriteBoolean("bias", false);
        }
    }

    /// <summary>The version of a network's part that can say this convolution: the fifth where it has no bias, otherwise what its window and groups need.</summary>
    /// <param name="withoutIt">The version the convolution needs for its window and its groups.</param>
    /// <returns>The version.</returns>
    private protected int SavedAs(int withoutIt) => Bias is null ? NetworkDocument.FifthVersion : withoutIt;

    // Channels a convolution can hold: one at least of each, in groups that divide both and no more than the channels in.
    private static Channels RequireChannels(Channels channels, ISpatialWalk walk)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(channels.In, 1, nameof(channels));
        ArgumentOutOfRangeException.ThrowIfLessThan(channels.Out, 1, nameof(channels));
        ArgumentOutOfRangeException.ThrowIfLessThan(channels.Groups, 1, nameof(channels));

        if (channels.In % channels.Groups != 0 || channels.Out % channels.Groups != 0)
        {
            // Two groups at least: one group divides every number of channels, so this is never said of one.
            throw new ArgumentException(
                $"A convolution through a {walk.Describe()} reads one channel at least and makes one at least, in groups that "
                + $"divide both; {channels.In} channels in and {channels.Out} out do not go into {channels.Groups} groups.",
                nameof(channels));
        }

        return channels;
    }

    /// <summary>
    /// What a convolution starts from: how its window walks, and — where the channels are stated rather than read off a
    /// kernel — those channels and the shape the kernel must then be.
    /// </summary>
    /// <remarks>
    /// The channels and the shape are stated together or not at all, since the one gives the other, so only
    /// <see cref="Taking"/> and <see cref="TransposedConvolution.GivingBack"/> make one: a family says how it lays a kernel
    /// out where every family can see it, rather than through a member a half-built convolution would have to ask for.
    /// </remarks>
    /// <param name="Walk">How its window walks.</param>
    /// <param name="Channels">The channels it was told of; nothing where they are to be read off the kernel.</param>
    /// <param name="Kernel">The shape its kernel must be; nothing where the kernel says the channels itself.</param>
    private protected readonly record struct Starting(ISpatialWalk Walk, Channels? Channels, Shape? Kernel);
}

/// <summary>How many channels a convolution reads and makes, and how many groups they are split into.</summary>
/// <remarks>
/// Each is said by name, because a number of channels in and a number of channels out are the same kind of number and swapping
/// them would be a convolution that compiles and is wrong: <c>new Channels { In = 4, Out = 6, Groups = 2 }</c>. In so many
/// groups, each group of channels in reaches its own group of channels out and no other, which is PyTorch's and ONNX's
/// <c>groups</c> and Keras's <c>groups</c>; as many groups as channels in is a depthwise convolution, where the channels out
/// are the channels in times Keras's <c>depth_multiplier</c>.
/// </remarks>
public readonly record struct Channels
{
    /// <summary>Channels in every group, unless said: every channel in reaches every channel out.</summary>
    public Channels()
    {
    }

    /// <summary>How many channels each place of an example holds.</summary>
    public required int In { get; init; }

    /// <summary>How many channels the convolution makes of each place.</summary>
    public required int Out { get; init; }

    /// <summary>How many groups the channels are split into; one unless said.</summary>
    public int Groups { get; init; } = 1;

    /// <inheritdoc />
    public override string ToString() =>
        Groups == 1 ? $"{In} channels in, {Out} out" : $"{In} channels in, {Out} out, in {Groups} groups";
}

/// <summary>
/// A convolution along a series: a window of so many steps slid over each, every place it stands given a value for each
/// channel the layer makes.
/// </summary>
/// <remarks>
/// A series is laid out step by channel, as Keras's <c>Conv1D</c> lays it out. Its window pads each series by the border it
/// states at each end, as TensorFlow's 'same' does, or — as Keras's 'causal' does — before the series alone, so no place sees
/// what comes after its own.
/// </remarks>
public sealed class Conv1D : Convolution, ISaved<Conv1D>
{
    /// <summary>A convolution that starts as PyTorch's does.</summary>
    /// <param name="inChannels">How many channels each step of a series holds.</param>
    /// <param name="outChannels">How many channels the layer makes of each place its window stands.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere: a length or a stride below one, or a border below nothing.</exception>
    public Conv1D(int inChannels, int outChannels, Window1D window, Draws draws)
        : this(new Channels { In = inChannels, Out = outChannels }, window, draws)
    {
    }

    /// <summary>A convolution of channels in groups, that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, or the window cannot stand anywhere.</exception>
    public Conv1D(Channels channels, Window1D window, Draws draws)
        : this(channels, window, new Beginning(draws))
    {
    }

    /// <summary>A convolution of channels in groups, that starts as it is told to.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="beginning">The draws its start is taken from, and how its kernel and its bias start — or that it has no bias.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere, or the beginning wants no bias and says how it starts.</exception>
    /// <exception cref="ArgumentNullException">The beginning holds no draws.</exception>
    public Conv1D(Channels channels, Window1D window, Beginning beginning)
        : base(Taking(new LineWalk(window), channels), beginning) => Window = window;

    /// <summary>A convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values, channels in included, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">
    /// The kernel is not a matrix whose rows are the window's length times a whole number of channels, the bias is not as long
    /// as the kernel is wide, or the window cannot stand anywhere.
    /// </exception>
    public Conv1D(Tensor kernel, Tensor? bias, Window1D window)
        : base(Taking(new LineWalk(window), null), kernel, bias) => Window = window;

    /// <summary>A convolution of channels in groups, that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values times the channels in of one group, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <exception cref="ArgumentException">The kernel is not the shape the window and the channels ask for, or the window cannot stand anywhere.</exception>
    public Conv1D(Tensor kernel, Tensor? bias, Window1D window, Channels channels)
        : base(Taking(new LineWalk(window), channels), kernel, bias) => Window = window;

    /// <summary>The patch, and how it walks.</summary>
    public Window1D Window { get; }

    /// <inheritdoc />
    public static string Name => "conv1d";

    /// <inheritdoc />
    /// <remarks>Its padding is the number of steps at each end, the word <c>same</c>, or the word <c>causal</c>.</remarks>
    public static Conv1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(
            ChannelsIn(settings, rebuilding),
            rebuilding.Window1DIn(settings),
            BeginningIn(settings, rebuilding));
    }

    /// <inheritdoc />
    public int FileVersion => SavedAs(Groups == 1 ? Window.FileVersion : NetworkDocument.FourthVersion);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteChannels(writer);

        writer.WriteWindow(Window);
    }
}

/// <summary>
/// A convolution through a volume: a block of planes, rows and columns slid through each, every place it stands given a
/// value for each channel the layer makes.
/// </summary>
/// <remarks>A volume is laid out plane by row by column by channel, as Keras's <c>Conv3D</c> lays it out.</remarks>
public sealed class Conv3D : Convolution, ISaved<Conv3D>
{
    /// <summary>A convolution that starts as PyTorch's does.</summary>
    /// <param name="inChannels">How many channels each place of a volume holds.</param>
    /// <param name="outChannels">How many channels the layer makes of each place its window stands.</param>
    /// <param name="window">The block, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere: a side or a stride below one, or a border below nothing.</exception>
    public Conv3D(int inChannels, int outChannels, Window3D window, Draws draws)
        : this(new Channels { In = inChannels, Out = outChannels }, window, draws)
    {
    }

    /// <summary>A convolution of channels in groups, that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, or the window cannot stand anywhere.</exception>
    public Conv3D(Channels channels, Window3D window, Draws draws)
        : this(channels, window, new Beginning(draws))
    {
    }

    /// <summary>A convolution of channels in groups, that starts as it is told to.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="beginning">The draws its start is taken from, and how its kernel and its bias start — or that it has no bias.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, the window cannot stand anywhere, or the beginning wants no bias and says how it starts.</exception>
    /// <exception cref="ArgumentNullException">The beginning holds no draws.</exception>
    public Conv3D(Channels channels, Window3D window, Beginning beginning)
        : base(Taking(new VolumeWalk(window), channels), beginning) => Window = window;

    /// <summary>A convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values, channels in included, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The block, and how it walks.</param>
    /// <exception cref="ArgumentException">
    /// The kernel is not a matrix whose rows are the window's places times a whole number of channels, the bias is not as long
    /// as the kernel is wide, or the window cannot stand anywhere.
    /// </exception>
    public Conv3D(Tensor kernel, Tensor? bias, Window3D window)
        : base(Taking(new VolumeWalk(window), null), kernel, bias) => Window = window;

    /// <summary>A convolution of channels in groups, that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values times the channels in of one group, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <exception cref="ArgumentException">The kernel is not the shape the window and the channels ask for, or the window cannot stand anywhere.</exception>
    public Conv3D(Tensor kernel, Tensor? bias, Window3D window, Channels channels)
        : base(Taking(new VolumeWalk(window), channels), kernel, bias) => Window = window;

    /// <summary>The block, and how it walks.</summary>
    public Window3D Window { get; }

    /// <inheritdoc />
    public static string Name => "conv3d";

    /// <inheritdoc />
    /// <remarks>Its padding is the number of planes, rows and columns on every side, the word <c>same</c>, or the word <c>causal</c>.</remarks>
    public static Conv3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(
            ChannelsIn(settings, rebuilding),
            rebuilding.Window3DIn(settings),
            BeginningIn(settings, rebuilding));
    }

    /// <inheritdoc />
    public int FileVersion => SavedAs(Groups == 1 ? Window.FileVersion : NetworkDocument.FourthVersion);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteChannels(writer);

        writer.WriteWindow(Window);
    }
}
