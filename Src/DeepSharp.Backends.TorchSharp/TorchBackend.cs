// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DeepSharp.Tensors;
using TorchSharp;

namespace DeepSharp.Backends.TorchSharp;

/// <summary>
/// A network's arithmetic on libtorch, through TorchSharp: on this machine's processor, <see cref="OnCpu()"/>, or on a
/// graphics card, <see cref="OnGpu(int)"/>, at full width or keeping its work in half of it.
/// </summary>
/// <remarks>
/// It is handed to whatever asks for arithmetic, as any engine is — to a run as <c>new FitOptions(seed) { Backend =
/// TorchBackend.OnCpu() }</c>, which trains the network, judges it by its validation rows and takes its report's measures on
/// it, and to a trained network as it serves, <c>trained.Predict(rows, engine)</c> — and nothing keeps it: the network
/// holds none and its file names none, so a network trained on it is read back and served on any other. A checkpoint of a
/// run on it names it, the libtorch it runs on, its device and its width, and a run goes on from that checkpoint only on an
/// engine made the same way — on the same device, at the same width, with the same libtorch.
/// <para>
/// Every operation asks first what the light engine asks of its tensors, so it refuses what the light engine refuses, in
/// the same words, where libtorch would stretch a column over a row or answer the mean of nothing with something that is not
/// a number. What it works out stays where libtorch keeps it, on the processor or the card, and a tensor from anywhere else —
/// the rows of a batch, a network's starting numbers, another device's — is taken in once, the first time it is handed
/// over, and kept for as long as that tensor lives. Gradients are worked out as they are on every engine, by the recording's
/// one rulebook, on libtorch's arithmetic; libtorch's own history of operations is never switched on. What it holds is let go of
/// by the .NET collector, so an allocation that fails for want of room while the card holds garbage the collector has not reached
/// is asked for again, once, after a collection.
/// </para>
/// <para>
/// An engine at <see cref="Precision.BFloat16"/> or <see cref="Precision.Half"/> keeps what it works out in that width, by one
/// table and nothing else: the products, and everything that follows a product (the row a layer adds, the rectified, the
/// sigmoid), are made and kept in the narrow width, and the numbers a network learns stay at full width, taken in at the narrow
/// one where a product reads them. What is added to or multiplied by a full-width tensor, and everything that must not lose its
/// digits — the exponent, the logarithm, a root, a share of a whole, a mean, a total — is worked out at full width and made at
/// full width. A value read back is the float that holds the narrow value exactly. The engine never sets libtorch's flags for
/// products at a lower width than the one asked for.
/// </para>
/// <para>
/// Its arithmetic is held to the light engine's operation by operation and step by step, not run by run: libtorch adds its
/// totals up in single precision and in its own order, so two runs of many steps drift apart, as any two float engines do.
/// It never sets how many threads libtorch works with, which is the application's, nor draws a random number: every draw a
/// run makes is DeepSharp's own.
/// </para>
/// </remarks>
public sealed class TorchBackend : INamesItsVersionAndDevice, INamesItsPrecision, IMultipliesInBlocks, IWidens
{
    private readonly torch.Device _device;
    private readonly string _owner;
    private readonly string _named;
    private readonly Precision _precision;
    private readonly torch.ScalarType _kept;
    private readonly ConditionalWeakTable<Tensor, TorchStorage> _takenInFull = new();
    private readonly ConditionalWeakTable<Tensor, TorchStorage> _takenInKept = new();
    private readonly ConditionalWeakTable<Tensor, TorchStorage>.CreateValueCallback _takeInFull;
    private readonly ConditionalWeakTable<Tensor, TorchStorage>.CreateValueCallback _takeInKept;

    private TorchBackend(torch.Device device, Precision precision)
    {
        _device = device;
        _owner = device.ToString();
        _named = _owner;
        _precision = precision;
        _kept = precision.Kept;
        _takeInFull = tensor => TakeIn(tensor, torch.ScalarType.Float32);
        _takeInKept = tensor => TakeIn(tensor, _kept);
    }

    /// <inheritdoc />
    /// <remarks>The same on the processor and on a card, at any width.</remarks>
    public string Name => "torch";

    /// <inheritdoc />
    /// <remarks>
    /// The libtorch it runs on, as TorchSharp states it — 2.10.0.0 for TorchSharp 0.107.0 — read from the TorchSharp the
    /// application runs, so a newer one an application brings is named as what it is.
    /// </remarks>
    public string Version => torch.__version__;

    /// <inheritdoc />
    /// <remarks>The device it was made for, as libtorch names it: <c>cpu</c>, or <c>cuda:0</c> for the first card.</remarks>
    public string Device => _named;

    /// <summary>The width this engine keeps what it works out in: full, unless it was made to keep it in half.</summary>
    public Precision Precision => _precision;

    /// <inheritdoc />
    /// <remarks>Nothing at full width, which names none, as every engine named itself before; <c>bfloat16</c> or <c>half</c> otherwise.</remarks>
    string? INamesItsPrecision.Precision =>
        _precision switch
        {
            Precision.Full => null,
            Precision.BFloat16 => "bfloat16",
            _ => "half",
        };

    /// <summary>An engine on this machine's processor, at full width.</summary>
    /// <returns>The engine.</returns>
    /// <exception cref="InvalidOperationException">
    /// The application brings no libtorch: the packages that bring one — the processor's for its platform, TorchSharp-cpu,
    /// or a graphics card's — are named.
    /// </exception>
    public static TorchBackend OnCpu() => OnCpu(Precision.Full);

    /// <summary>An engine on this machine's processor, keeping what it works out in the given width.</summary>
    /// <param name="precision">The width; <see cref="Precision.Half"/> is a card's and is refused here.</param>
    /// <returns>The engine.</returns>
    /// <exception cref="ArgumentException">The width is half, which a processor's libtorch has no hardware for, so it takes about six hundred times as long for a product as at full width; the instructions it needs and the width to use instead are named.</exception>
    /// <exception cref="InvalidOperationException">The application brings no libtorch.</exception>
    public static TorchBackend OnCpu(Precision precision) =>
        precision == Precision.Half
            ? throw Libtorch.NoHalfOnTheProcessor()
            : new(Libtorch.Loaded(() => torch.CPU), precision);

    /// <summary>An engine on a graphics card, at full width.</summary>
    /// <param name="index">Which card, numbered from nought as libtorch numbers them.</param>
    /// <returns>The engine.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The number is below nothing, or there is no card of that number here.</exception>
    /// <exception cref="InvalidOperationException">
    /// The application brings no libtorch, or libtorch finds no card it can run on here — the libtorch brought is the
    /// processor's, or the machine has no NVIDIA card with its driver; the packages that bring a card's libtorch are named.
    /// </exception>
    public static TorchBackend OnGpu(int index) => OnGpu(index, Precision.Full);

    /// <summary>An engine on a graphics card, keeping what it works out in the given width.</summary>
    /// <param name="index">Which card, numbered from nought as libtorch numbers them.</param>
    /// <param name="precision">The width.</param>
    /// <returns>The engine.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The number is below nothing, or there is no card of that number here.</exception>
    /// <exception cref="InvalidOperationException">The application brings no libtorch, or libtorch finds no card it can run on here.</exception>
    public static TorchBackend OnGpu(int index, Precision precision)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        var cards = Libtorch.Loaded(torch.cuda.device_count);

        return index < cards ? new(torch.device(DeviceType.CUDA, index), precision) : throw Libtorch.NoCard(index, cards);
    }

    /// <inheritdoc />
    /// <remarks>The values are taken in at full width, once, and kept for as long as the tensor lives; a tensor at full width already is handed back as it is.</remarks>
    public Tensor Widened(Tensor values)
    {
        ArgumentNullException.ThrowIfNull(values);

        var held = Held(values, torch.ScalarType.Float32);

        return ReferenceEquals(held, values.Storage) ? values : Tensor.On(values.Shape, held);
    }

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Add));

        return Meeting(left, right, left.Shape, static (a, b) => a.add(b));
    }

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Subtract));

        return Meeting(left, right, left.Shape, static (a, b) => a.sub(b));
    }

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Multiply));

        return Meeting(left, right, left.Shape, static (a, b) => a.mul(b));
    }

    /// <inheritdoc />
    /// <remarks>Worked out at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Divide(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Divide));

        return Both(new Reading(left, torch.ScalarType.Float32), new Reading(right, torch.ScalarType.Float32), left.Shape, static (a, b) => a.div(b));
    }

    /// <inheritdoc />
    /// <remarks>Both tensors are read at the width the engine keeps its products in, and the product is made and kept in it.</remarks>
    public Tensor MatMul(Tensor left, Tensor right)
    {
        left.RequireMatrixProduct(right);

        return Both(new Reading(left, _kept), new Reading(right, _kept), new Shape(left.Shape[0], right.Shape[1]), static (a, b) => a.mm(b));
    }

    /// <inheritdoc />
    /// <remarks>One batched product of libtorch's, the blocks laid out as its batch, at the width the engine keeps its products in.</remarks>
    Tensor IMultipliesInBlocks.MatMulInBlocks(Tensor left, Tensor right, BlockProduct how)
    {
        var sizes = left.RequireBlockProduct(right, how);

        return Both(
            new Reading(left, _kept),
            new Reading(right, _kept),
            new Shape(how.Count * sizes.Rows, sizes.Columns),
            (first, second) => OfBlocks(first, sizes.Left, how.Count, how.TurnLeft).bmm(OfBlocks(second, sizes.Right, how.Count, how.TurnRight)).reshape(how.Count * sizes.Rows, sizes.Columns));
    }

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(Transpose));

        return One(matrix, FormatOf(matrix), new Shape(matrix.Shape[1], matrix.Shape[0]), static values => values.t().contiguous());
    }

    /// <inheritdoc />
    /// <remarks>The row is read at the width the matrix is in, so a bias added to a product leaves it where the product was.</remarks>
    public Tensor AddRow(Tensor matrix, Tensor row)
    {
        matrix.RequireRow(row);

        var format = FormatOf(matrix);

        return Both(new Reading(matrix, format), new Reading(row, format), matrix.Shape, static (values, added) => values.add(added));
    }

    /// <inheritdoc />
    /// <remarks>Totalled at full width, whatever width the engine keeps its work in, from the values as they are held: no copy of them at full width is made.</remarks>
    public Tensor SumRows(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(SumRows));

        // The axis as a long: a bare nought would be taken for a type of number to add up in.
        return One(matrix, FormatOf(matrix), new Shape(matrix.Shape[1]), static values => values.sum(0L, false, torch.ScalarType.Float32));
    }

    /// <inheritdoc />
    /// <remarks>Averaged at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Mean(Tensor values)
    {
        values.RequireValues();

        return One(values, torch.ScalarType.Float32, new Shape(), static held => held.mean());
    }

    /// <inheritdoc />
    /// <remarks>The factor is one value and is read at full width; the values keep the width they were in.</remarks>
    public Tensor Scale(Tensor values, Tensor factor)
    {
        values.RequireFactor(factor);

        return Both(new Reading(values, FormatOf(values)), new Reading(factor, torch.ScalarType.Float32), values.Shape, static (held, by) => held.mul(by));
    }

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value)
    {
        using (torch.NewDisposeScope())
        {
            return Made(shape, WithRoom(() => torch.full(shape.Lengths(), value, torch.ScalarType.Float32, _device)));
        }
    }

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => Each(values, static held => held.relu());

    /// <inheritdoc />
    /// <remarks>The step is made at the width of the values it was taken of.</remarks>
    public Tensor Positive(Tensor values) => Each(values, static held => held.gt(0f).to_type(held.dtype));

    /// <inheritdoc />
    public Tensor Tanh(Tensor values) => Each(values, static held => held.tanh());

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => Each(values, static held => held.sigmoid());

    /// <inheritdoc />
    /// <remarks>Worked out at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Exp(Tensor values) => Wide(values, static held => held.exp());

    /// <inheritdoc />
    /// <remarks>Worked out at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Log(Tensor values) => Wide(values, static held => held.log());

    /// <inheritdoc />
    /// <remarks>Worked out at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Sqrt(Tensor values) => Wide(values, static held => held.sqrt());

    /// <inheritdoc />
    /// <remarks>As libtorch has it: the value itself above twenty, the logarithm of one plus its exponential below; at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Softplus(Tensor values) => Wide(values, static held => held.softplus());

    /// <inheritdoc />
    /// <remarks>Worked out at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Sin(Tensor values) => Wide(values, static held => held.sin());

    /// <inheritdoc />
    /// <remarks>Worked out at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor Erf(Tensor values) => Wide(values, static held => held.erf());

    /// <inheritdoc />
    /// <remarks>The shares of a row are worked out at full width, whatever width the engine keeps its work in.</remarks>
    public Tensor LogSoftmax(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(LogSoftmax));

        return One(matrix, torch.ScalarType.Float32, matrix.Shape, static values => values.log_softmax(1));
    }

    /// <inheritdoc />
    /// <remarks>libtorch's argmax, which picks the first of values that tie and takes a value that is not a number as the largest; made at the width of the values.</remarks>
    public Tensor FirstLargest(Tensor matrix)
    {
        matrix.RequireRowsToPickFrom(nameof(FirstLargest));

        return One(matrix, FormatOf(matrix), matrix.Shape, static values => torch.nn.functional.one_hot(values.argmax(1), values.shape[1]).to_type(values.dtype));
    }

    /// <inheritdoc />
    /// <remarks>Stands on the storage it was handed, as a view of it: the values are the same values, in their order.</remarks>
    public Tensor Reshape(Tensor values, Shape shape)
    {
        values.RequireSameCount(shape);

        return Tensor.On(shape, Held(values, FormatOf(values)));
    }

    /// <inheritdoc />
    /// <remarks>
    /// libtorch unfolds images laid out channel by channel, each patch's values a channel at a time, so the images are turned
    /// round for it and every patch turned back: each place's channels side by side, as the seam lays a patch out.
    /// </remarks>
    public Tensor Unfold(Tensor images, Window window)
    {
        images.RequireImagesFor(window);

        var walk = new WindowWalk(images.Shape, window);

        return One(images, FormatOf(images), walk.Patches, walk.Unfolded);
    }

    /// <inheritdoc />
    /// <remarks>Unfolding turned around: the patches turned into libtorch's layout, folded onto the images with their border, and the border cut away.</remarks>
    public Tensor Fold(Tensor patches, Shape images, Window window)
    {
        patches.RequirePatchesOf(images, window);

        return One(patches, FormatOf(patches), images, new WindowWalk(images, window).Folded);
    }

    // A stacked matrix as libtorch's batch of matrices, each turned on its side where it is to be.
    private static torch.Tensor OfBlocks(torch.Tensor stack, BlockShape block, int count, bool turned)
    {
        var blocks = stack.reshape(count, block.Rows, block.Columns);

        return turned ? blocks.transpose(1, 2) : blocks;
    }

    // The width a tensor is read in as it stands: the width this engine keeps its work in, for a tensor of its own at that width,
    // and full width for any other — an engine at full width reads every tensor so.
    private torch.ScalarType FormatOf(Tensor tensor) =>
        tensor.Storage is TorchStorage own && own.IsHeldFor(_owner, _kept) ? _kept : torch.ScalarType.Float32;

    // A tensor's values where this engine works on them, in the width asked for: its own storage on its own device in that width as
    // it is, any other taken in once and kept for as long as the tensor lives.
    private TorchStorage Held(Tensor tensor, torch.ScalarType format) =>
        tensor.Storage is TorchStorage own && own.IsHeldFor(_owner, format)
            ? own
            : WithRoom(() => format == torch.ScalarType.Float32 ? _takenInFull.GetValue(tensor, _takeInFull) : _takenInKept.GetValue(tensor, _takeInKept));

    // An operation run in a scope of its own, so that what a try that failed had made is let go of before the next try and what a good try made is
    // handed to the scope around it.
    private static torch.Tensor Tried(Func<torch.Tensor> operation) =>
        WithRoom(() =>
        {
            using var scope = torch.NewDisposeScope();

            return operation().MoveToOuterDisposeScope();
        });

    // What libtorch holds is let go of by the collector, which runs when it is told of enough memory to run for and not when the card is full: an
    // allocation that fails while the card holds garbage the collector has not reached is asked for again once the collector has been through.
    internal static T WithRoom<T>(Func<T> allocate)
    {
        try
        {
            return allocate();
        }
        catch (Exception fault) when (fault.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();

            return allocate();
        }
    }

    // A tensor on another device or in another width, moved to this device and cast; a tensor on any other storage, copied into
    // libtorch's memory and moved to this device in the width asked for: kept for as long as the tensor lives.
    private TorchStorage TakeIn(Tensor tensor, torch.ScalarType format)
    {
        using (torch.NewDisposeScope())
        {
            if (tensor.Storage is TorchStorage elsewhere)
            {
                var moved = new TorchStorage(elsewhere.As(tensor.Shape).to_type(format).to(_device), _owner);
                GC.KeepAlive(elsewhere);

                return moved;
            }

            var host = torch.empty(tensor.Shape.Lengths(), torch.ScalarType.Float32);
            tensor.Storage.CopyTo(MemoryMarshal.Cast<byte, float>(host.bytes));

            return new TorchStorage(host.to(format, _device), _owner);
        }
    }

    // What an operation worked out, on a storage of this engine's that owns it.
    private Tensor Made(Shape shape, torch.Tensor native) => Tensor.On(shape, new TorchStorage(native, _owner));

    // A value-by-value operation of one tensor that leaves it at the width it was in, whose refusal is only that it is handed.
    private Tensor Each(Tensor values, Func<torch.Tensor, torch.Tensor> operation)
    {
        ArgumentNullException.ThrowIfNull(values);

        return One(values, FormatOf(values), values.Shape, operation);
    }

    // A value-by-value operation of one tensor that is worked out at full width, whatever width the values were in.
    private Tensor Wide(Tensor values, Func<torch.Tensor, torch.Tensor> operation)
    {
        ArgumentNullException.ThrowIfNull(values);

        return One(values, torch.ScalarType.Float32, values.Shape, operation);
    }

    // An operation of one tensor read in the width asked for, in a dispose scope of its own: whatever libtorch makes on the way is
    // let go of at its end, and what it worked out is taken out of the scope by the storage that owns it. The tensor read is kept
    // alive until libtorch has read it.
    private Tensor One(Tensor values, torch.ScalarType format, Shape shape, Func<torch.Tensor, torch.Tensor> operation)
    {
        using (torch.NewDisposeScope())
        {
            var held = Held(values, format);
            var made = Made(shape, Tried(() => operation(held.As(values.Shape))));

            GC.KeepAlive(held);

            return made;
        }
    }

    // An operation of two tensors that meet: both are read at the width they share, and at full width when they do not.
    private Tensor Meeting(Tensor left, Tensor right, Shape shape, Func<torch.Tensor, torch.Tensor, torch.Tensor> operation)
    {
        var format = FormatOf(left) == FormatOf(right) ? FormatOf(left) : torch.ScalarType.Float32;

        return Both(new Reading(left, format), new Reading(right, format), shape, operation);
    }

    // An operation of two tensors, each read in the width asked for, likewise.
    private Tensor Both(Reading left, Reading right, Shape shape, Func<torch.Tensor, torch.Tensor, torch.Tensor> operation)
    {
        using (torch.NewDisposeScope())
        {
            var first = Held(left.Tensor, left.Format);
            var second = Held(right.Tensor, right.Format);
            var made = Made(shape, Tried(() => operation(first.As(left.Tensor.Shape), second.As(right.Tensor.Shape))));

            GC.KeepAlive(first);
            GC.KeepAlive(second);

            return made;
        }
    }

    // A tensor and the width an operation reads it in.
    private readonly record struct Reading(Tensor Tensor, torch.ScalarType Format);

    /// <summary>
    /// A window's walk over a batch of images, channels last, in libtorch's terms: the places it stands, the border it pads
    /// each side with and how far apart its cells are — as <see cref="Window.PlacingOver"/> works them out, the one rule every
    /// engine reads — and the patches it takes.
    /// </summary>
    /// <remarks>
    /// libtorch counts the places a window fits whole, where a window of this library may be asked to stand at one more whose
    /// patch overhangs the end. The end is therefore given as much more border as that last patch needs, so libtorch's own
    /// count is this window's count, and the same is narrowed off again after folding: what the patch covers beyond the image
    /// is border either way.
    /// </remarks>
    private sealed class WindowWalk
    {
        private readonly Window _window;
        private readonly Placing _placing;
        private readonly long _count;
        private readonly long _height;
        private readonly long _width;
        private readonly long _channels;
        private readonly long _places;
        private readonly long _beyondDown;
        private readonly long _beyondAcross;

        public WindowWalk(Shape images, Window window)
        {
            _window = window;
            _placing = window.PlacingOver(images[1], images[2]);
            _count = images[0];
            _height = images[1];
            _width = images[2];
            _channels = images[3];

            long rows = window.RowsOver(images[1]);
            long columns = window.ColumnsOver(images[2]);

            _places = rows * columns;
            _beyondDown = Beyond(rows, _placing.Stride.Down, Reach(window.Height, _placing.Dilation.Down), _height + _placing.Border.Top + _placing.Border.Bottom);
            _beyondAcross = Beyond(columns, _placing.Stride.Across, Reach(window.Width, _placing.Dilation.Across), _width + _placing.Border.Left + _placing.Border.Right);
            Patches = new Shape(checked((int)(_count * _places)), window.Height * window.Width * images[3]);
        }

        /// <summary>The patches' shape: a row for every image and every place the window stands, each as long as the window holds.</summary>
        public Shape Patches { get; }

        /// <summary>The patches of the images, channels last, as the seam lays them out.</summary>
        public torch.Tensor Unfolded(torch.Tensor images)
        {
            // Channels first, with the border on every side that has one, as libtorch unfolds.
            var padded = torch.nn.functional.pad(
                images.permute(0, 3, 1, 2),
                (_placing.Border.Left, _placing.Border.Right + _beyondAcross, _placing.Border.Top, _placing.Border.Bottom + _beyondDown));
            var columns = torch.nn.functional.unfold(
                padded,
                (_window.Height, _window.Width),
                dilation: (_placing.Dilation.Down, _placing.Dilation.Across),
                stride: (_placing.Stride.Down, _placing.Stride.Across));

            // libtorch lays a patch out channel by channel; here a place's channels stand side by side.
            return columns.reshape(_count, _channels, _window.Height, _window.Width, _places)
                .permute(0, 4, 2, 3, 1)
                .reshape(_count * _places, _window.Height * _window.Width * _channels)
                .contiguous();
        }

        /// <summary>The patches folded back onto the images, channels last, what fell on the border cut away.</summary>
        public torch.Tensor Folded(torch.Tensor patches)
        {
            var columns = patches.reshape(_count, _places, _window.Height, _window.Width, _channels)
                .permute(0, 4, 2, 3, 1)
                .reshape(_count, _channels * _window.Height * _window.Width, _places);
            var folded = torch.nn.functional.fold(
                columns,
                (_height + _placing.Border.Top + _placing.Border.Bottom + _beyondDown,
                 _width + _placing.Border.Left + _placing.Border.Right + _beyondAcross),
                (_window.Height, _window.Width),
                dilation: (_placing.Dilation.Down, _placing.Dilation.Across),
                stride: (_placing.Stride.Down, _placing.Stride.Across));

            return folded.narrow(2, _placing.Border.Top, _height).narrow(3, _placing.Border.Left, _width).permute(0, 2, 3, 1).contiguous();
        }

        // How many places of an axis the window spans, the gaps of a dilated window included.
        private static long Reach(int size, int dilation) => ((long)dilation * (size - 1)) + 1;

        // How much more border the end of an axis needs for libtorch's own count of whole-fitting places to be this window's.
        private static long Beyond(long places, int stride, long reach, long padded) =>
            Math.Max(0, ((places - 1) * stride) + reach - padded);
    }
}
