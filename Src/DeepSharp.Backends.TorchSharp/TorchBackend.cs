// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DeepSharp.Tensors;
using TorchSharp;

namespace DeepSharp.Backends.TorchSharp;

/// <summary>
/// A network's arithmetic on libtorch, through TorchSharp: on this machine's processor, <see cref="OnCpu"/>, or on a
/// graphics card, <see cref="OnGpu"/>.
/// </summary>
/// <remarks>
/// It is handed to whatever asks for arithmetic, as any engine is — to a run as <c>new FitOptions(seed) { Backend =
/// TorchBackend.OnCpu() }</c>, which trains the network, judges it by its validation rows and takes its report's measures on
/// it, and to a trained network as it serves, <c>trained.Predict(rows, engine)</c> — and nothing keeps it: the network
/// holds none and its file names none, so a network trained on it is read back and served on any other. A checkpoint of a
/// run on it names it, the libtorch it runs on and its device, and a run goes on from that checkpoint only on an engine made
/// the same way — on the same device, with the same libtorch.
/// <para>
/// Every operation asks first what the light engine asks of its tensors, so it refuses what the light engine refuses, in
/// the same words, where libtorch would stretch a column over a row or answer the mean of nothing with something that is not
/// a number. What it works out stays where libtorch keeps it, on the processor or the card, and a tensor from anywhere else —
/// the rows of a batch, a network's starting numbers, another device's — is taken in once, the first time it is handed
/// over, and kept for as long as that tensor lives. Gradients are worked out as they are on every engine, by the recording's
/// one rulebook, on libtorch's arithmetic; libtorch's own history of operations is never switched on.
/// </para>
/// <para>
/// Its arithmetic is held to the light engine's operation by operation and step by step, not run by run: libtorch adds its
/// totals up in single precision and in its own order, so two runs of many steps drift apart, as any two float engines do.
/// It never sets how many threads libtorch works with, which is the application's, nor draws a random number: every draw a
/// run makes is DeepSharp's own.
/// </para>
/// </remarks>
public sealed class TorchBackend : INamesItsVersionAndDevice
{
    private readonly torch.Device _device;
    private readonly string _named;
    private readonly ConditionalWeakTable<Tensor, TorchStorage> _takenIn = new();
    private readonly ConditionalWeakTable<Tensor, TorchStorage>.CreateValueCallback _takeIn;

    private TorchBackend(torch.Device device)
    {
        _device = device;
        _named = device.ToString();
        _takeIn = TakeIn;
    }

    /// <inheritdoc />
    /// <remarks>The same on the processor and on a card.</remarks>
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

    /// <summary>An engine on this machine's processor.</summary>
    /// <returns>The engine.</returns>
    /// <exception cref="InvalidOperationException">
    /// The application brings no libtorch: the packages that bring one — the processor's for its platform, TorchSharp-cpu,
    /// or a graphics card's — are named.
    /// </exception>
    public static TorchBackend OnCpu() => new(Libtorch.Loaded(() => torch.CPU));

    /// <summary>An engine on a graphics card.</summary>
    /// <param name="index">Which card, numbered from nought as libtorch numbers them.</param>
    /// <returns>The engine.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The number is below nothing, or there is no card of that number here.</exception>
    /// <exception cref="InvalidOperationException">
    /// The application brings no libtorch, or libtorch finds no card it can run on here — the libtorch brought is the
    /// processor's, or the machine has no NVIDIA card with its driver; the packages that bring a card's libtorch are named.
    /// </exception>
    public static TorchBackend OnGpu(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        var cards = Libtorch.Loaded(torch.cuda.device_count);

        return index < cards ? new(torch.device(DeviceType.CUDA, index)) : throw Libtorch.NoCard(index, cards);
    }

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Add));

        return Both(left, right, left.Shape, static (a, b) => a.add(b));
    }

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Subtract));

        return Both(left, right, left.Shape, static (a, b) => a.sub(b));
    }

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Multiply));

        return Both(left, right, left.Shape, static (a, b) => a.mul(b));
    }

    /// <inheritdoc />
    public Tensor Divide(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Divide));

        return Both(left, right, left.Shape, static (a, b) => a.div(b));
    }

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right)
    {
        left.RequireMatrixProduct(right);

        return Both(left, right, new Shape(left.Shape[0], right.Shape[1]), static (a, b) => a.mm(b));
    }

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(Transpose));

        return One(matrix, new Shape(matrix.Shape[1], matrix.Shape[0]), static values => values.t().contiguous());
    }

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row)
    {
        matrix.RequireRow(row);

        return Both(matrix, row, matrix.Shape, static (values, added) => values.add(added));
    }

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(SumRows));

        // The axis as a long: a bare nought would be taken for a type of number to add up in.
        return One(matrix, new Shape(matrix.Shape[1]), static values => values.sum(0L));
    }

    /// <inheritdoc />
    public Tensor Mean(Tensor values)
    {
        values.RequireValues();

        return One(values, new Shape(), static held => held.mean());
    }

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor)
    {
        values.RequireFactor(factor);

        return Both(values, factor, values.Shape, static (held, by) => held.mul(by));
    }

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value)
    {
        using (torch.NewDisposeScope())
        {
            return Made(shape, torch.full(shape.Lengths(), value, torch.ScalarType.Float32, _device));
        }
    }

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => Each(values, static held => held.relu());

    /// <inheritdoc />
    public Tensor Positive(Tensor values) => Each(values, static held => held.gt(0f).to_type(torch.ScalarType.Float32));

    /// <inheritdoc />
    public Tensor Tanh(Tensor values) => Each(values, static held => held.tanh());

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => Each(values, static held => held.sigmoid());

    /// <inheritdoc />
    public Tensor Exp(Tensor values) => Each(values, static held => held.exp());

    /// <inheritdoc />
    public Tensor Log(Tensor values) => Each(values, static held => held.log());

    /// <inheritdoc />
    public Tensor Sqrt(Tensor values) => Each(values, static held => held.sqrt());

    /// <inheritdoc />
    /// <remarks>As libtorch has it: the value itself above twenty, the logarithm of one plus its exponential below.</remarks>
    public Tensor Softplus(Tensor values) => Each(values, static held => held.softplus());

    /// <inheritdoc />
    public Tensor LogSoftmax(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(LogSoftmax));

        return One(matrix, matrix.Shape, static values => values.log_softmax(1));
    }

    /// <inheritdoc />
    /// <remarks>Stands on the storage it was handed, as a view of it: the values are the same values, in their order.</remarks>
    public Tensor Reshape(Tensor values, Shape shape)
    {
        values.RequireSameCount(shape);

        return Tensor.On(shape, Held(values));
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

        return One(images, walk.Patches, walk.Unfolded);
    }

    /// <inheritdoc />
    /// <remarks>Unfolding turned around: the patches turned into libtorch's layout, folded onto the images with their border, and the border cut away.</remarks>
    public Tensor Fold(Tensor patches, Shape images, Window window)
    {
        patches.RequirePatchesOf(images, window);

        return One(patches, images, new WindowWalk(images, window).Folded);
    }

    // A tensor's values where this engine works on them: its own storage on its own device as it is, any other taken in once.
    private TorchStorage Held(Tensor tensor) =>
        tensor.Storage is TorchStorage own && own.IsOn(_named) ? own : _takenIn.GetValue(tensor, _takeIn);

    // A tensor on another device, moved to this one; a tensor on any other storage, copied into libtorch's memory and
    // moved to this device: kept for as long as the tensor lives.
    private TorchStorage TakeIn(Tensor tensor)
    {
        using (torch.NewDisposeScope())
        {
            if (tensor.Storage is TorchStorage elsewhere)
            {
                var moved = new TorchStorage(elsewhere.As(tensor.Shape).to(_device), _named);
                GC.KeepAlive(elsewhere);

                return moved;
            }

            var host = torch.empty(tensor.Shape.Lengths(), torch.ScalarType.Float32);
            tensor.Storage.CopyTo(MemoryMarshal.Cast<byte, float>(host.bytes));

            return new TorchStorage(host.to(_device), _named);
        }
    }

    // What an operation worked out, on a storage of this engine's that owns it.
    private Tensor Made(Shape shape, torch.Tensor native) => Tensor.On(shape, new TorchStorage(native, _named));

    // A value-by-value operation of one tensor, whose refusal is only that it is handed.
    private Tensor Each(Tensor values, Func<torch.Tensor, torch.Tensor> operation)
    {
        ArgumentNullException.ThrowIfNull(values);

        return One(values, values.Shape, operation);
    }

    // An operation of one tensor, in a dispose scope of its own: whatever libtorch makes on the way is let go of at its end,
    // and what it worked out is taken out of the scope by the storage that owns it. The tensor read is kept alive until
    // libtorch has read it.
    private Tensor One(Tensor values, Shape shape, Func<torch.Tensor, torch.Tensor> operation)
    {
        using (torch.NewDisposeScope())
        {
            var held = Held(values);
            var made = Made(shape, operation(held.As(values.Shape)));

            GC.KeepAlive(held);

            return made;
        }
    }

    // An operation of two tensors, likewise.
    private Tensor Both(Tensor left, Tensor right, Shape shape, Func<torch.Tensor, torch.Tensor, torch.Tensor> operation)
    {
        using (torch.NewDisposeScope())
        {
            var first = Held(left);
            var second = Held(right);
            var made = Made(shape, operation(first.As(left.Shape), second.As(right.Shape)));

            GC.KeepAlive(first);
            GC.KeepAlive(second);

            return made;
        }
    }

    /// <summary>
    /// A window's walk over a batch of images, channels last, in libtorch's terms: the places it stands, the border it pads
    /// each side with — as <see cref="Window.BordersOver"/> works it out, the one rule every engine reads — and the patches it
    /// takes.
    /// </summary>
    private sealed class WindowWalk
    {
        private readonly Window _window;
        private readonly Borders _borders;
        private readonly long _count;
        private readonly long _height;
        private readonly long _width;
        private readonly long _channels;
        private readonly long _places;

        public WindowWalk(Shape images, Window window)
        {
            _window = window;
            _borders = window.BordersOver(images[1], images[2]);
            _count = images[0];
            _height = images[1];
            _width = images[2];
            _channels = images[3];
            _places = (long)window.RowsOver(images[1]) * window.ColumnsOver(images[2]);
            Patches = new Shape(checked((int)(_count * _places)), window.Height * window.Width * images[3]);
        }

        /// <summary>The patches' shape: a row for every image and every place the window stands, each as long as the window holds.</summary>
        public Shape Patches { get; }

        /// <summary>The patches of the images, channels last, as the seam lays them out.</summary>
        public torch.Tensor Unfolded(torch.Tensor images)
        {
            // Channels first, with the border on every side that has one, as libtorch unfolds.
            var padded = torch.nn.functional.pad(images.permute(0, 3, 1, 2), (_borders.Left, _borders.Right, _borders.Top, _borders.Bottom));
            var columns = torch.nn.functional.unfold(padded, (_window.Height, _window.Width), stride: (_window.Stride, _window.Stride));

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
                (_height + _borders.Top + _borders.Bottom, _width + _borders.Left + _borders.Right),
                (_window.Height, _window.Width),
                stride: (_window.Stride, _window.Stride));

            return folded.narrow(2, _borders.Top, _height).narrow(3, _borders.Left, _width).permute(0, 2, 3, 1).contiguous();
        }
    }
}
