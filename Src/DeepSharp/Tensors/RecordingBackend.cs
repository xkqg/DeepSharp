// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tensors;

/// <summary>
/// A backend that writes down every operation of one pass as it runs it on the backend it wraps, and then works back
/// from a loss to how much each tensor the pass read moved it.
/// </summary>
/// <param name="inner">The backend the arithmetic runs on.</param>
/// <remarks>
/// Made for one pass and handed to what runs it, like any backend, and dropped once the gradients are taken: nothing
/// holds one, shares one or reaches for one, so a pass that wants gradients says so by being handed one. What it keeps
/// is a record of this pass's arithmetic, not a second description of the model — the model is whatever code ran.
/// <para>
/// Every operation the seam offers is written down with the rule that sends a gradient back through it, so an operation
/// added to the seam without such a rule does not compile. The way back runs on the wrapped backend's own operations
/// and reads no tensor's values, so it works wherever the arithmetic does. A tensor read more than once gets the sum
/// of what every reading sends back.
/// </para>
/// <para>
/// The way back is worked out only towards the tensors asked for, as libtorch records an operation only when one of the
/// tensors it reads wants a gradient: whatever the pass made of one of them leads back to it, and nothing else does. A
/// share that could reach none of them — for the rows of a batch, the answers a loss reads, a factor filled in — is never
/// worked out, so a convolution's way back folds nothing onto its images, and every gradient asked for comes to the same
/// bits it would if everything the pass read were asked for too.
/// </para>
/// </remarks>
public sealed class RecordingBackend(ITensorBackend inner) : ITensorBackend, IMultipliesInBlocks
{
    private readonly ITensorBackend _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    private readonly List<Recorded> _steps = [];

    /// <inheritdoc />
    public string Name => _inner.Name;

    /// <summary>The backend the arithmetic runs on, which records nothing.</summary>
    internal ITensorBackend Inner => _inner;

    /// <summary>How many operations this pass has written down so far.</summary>
    internal int Operations => _steps.Count;

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right) => Kept(new Added(left, right, _inner.Add(left, right)));

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right) => Kept(new Subtracted(left, right, _inner.Subtract(left, right)));

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right) => Kept(new Multiplied(left, right, _inner.Multiply(left, right)));

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right) => Kept(new MatrixProduct(left, right, _inner.MatMul(left, right)));

    /// <inheritdoc />
    /// <remarks>Written down as one product, whether the wrapped engine has it or the library works it out from the products the engine has.</remarks>
    Tensor IMultipliesInBlocks.MatMulInBlocks(Tensor left, Tensor right, BlockProduct how) =>
        Kept(new BlockedProduct(left, right, how, _inner.MatMulInBlocks(left, right, how)));

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix) => Kept(new Transposed(matrix, _inner.Transpose(matrix)));

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row) => Kept(new RowAdded(matrix, row, _inner.AddRow(matrix, row)));

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix) => Kept(new RowsSummed(matrix, _inner.SumRows(matrix)));

    /// <inheritdoc />
    public Tensor Mean(Tensor values) => Kept(new Averaged(values, _inner.Mean(values)));

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor) => Kept(new Scaled(values, factor, _inner.Scale(values, factor)));

    /// <inheritdoc />
    /// <remarks>A filled tensor reads nothing, so there is nothing to send a gradient back to.</remarks>
    public Tensor Fill(Shape shape, float value) => _inner.Fill(shape, value);

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => Kept(new Rectified(values, _inner.Relu(values)));

    /// <inheritdoc />
    public Tensor Positive(Tensor values) => Kept(new Stepped(values, _inner.Positive(values)));

    /// <inheritdoc />
    public Tensor FirstLargest(Tensor matrix) => Kept(new Stepped(matrix, _inner.FirstLargest(matrix)));

    /// <inheritdoc />
    public Tensor Tanh(Tensor values) => Kept(new TanhOf(values, _inner.Tanh(values)));

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => Kept(new SigmoidOf(values, _inner.Sigmoid(values)));

    /// <inheritdoc />
    public Tensor Exp(Tensor values) => Kept(new Exponentiated(values, _inner.Exp(values)));

    /// <inheritdoc />
    public Tensor Log(Tensor values) => Kept(new LogarithmOf(values, _inner.Log(values)));

    /// <inheritdoc />
    public Tensor Sqrt(Tensor values) => Kept(new SquareRootOf(values, _inner.Sqrt(values)));

    /// <inheritdoc />
    public Tensor Softplus(Tensor values) => Kept(new SoftplusOf(values, _inner.Softplus(values)));

    /// <inheritdoc />
    public Tensor Sin(Tensor values) => Kept(new SineOf(values, _inner.Sin(values)));

    /// <inheritdoc />
    public Tensor Erf(Tensor values) => Kept(new ErrorFunctionOf(values, _inner.Erf(values)));

    /// <inheritdoc />
    public Tensor Divide(Tensor left, Tensor right) => Kept(new Divided(left, right, _inner.Divide(left, right)));

    /// <inheritdoc />
    public Tensor LogSoftmax(Tensor matrix) => Kept(new LogSharesOf(matrix, _inner.LogSoftmax(matrix)));

    /// <inheritdoc />
    public Tensor Reshape(Tensor values, Shape shape) => Kept(new Reshaped(values, _inner.Reshape(values, shape)));

    /// <inheritdoc />
    public Tensor Unfold(Tensor images, Window window) => Kept(new Unfolded(images, window, _inner.Unfold(images, window)));

    /// <inheritdoc />
    public Tensor Fold(Tensor patches, Shape images, Window window) =>
        Kept(new Folded(patches, window, _inner.Fold(patches, images, window)));

    /// <summary>How much each of the given tensors moved a loss this pass worked out.</summary>
    /// <param name="loss">The loss: one value, with no axes, worked out in this pass.</param>
    /// <param name="parameters">The tensors whose gradients are wanted — the numbers a network learns — each one the pass read.</param>
    /// <returns>A gradient for each, of its own shape; nothing, as zeros, for one the pass read without it reaching the loss.</returns>
    /// <exception cref="ArgumentException">
    /// The loss has axes or was not worked out in this pass, or a tensor asked for is one the pass never read.
    /// </exception>
    /// <remarks>
    /// The loss's own gradient is one, and each operation from the loss back to the start sends what it received on to
    /// what it read — to what leads to a tensor asked for, and to nothing else. A tensor the pass never read is refused
    /// rather than given a gradient of nothing: asking for it is asking about a number the pass had no way to move, and
    /// that is a mistake in the asking.
    /// </remarks>
    public Gradients GradientsOf(Tensor loss, IEnumerable<Tensor> parameters)
    {
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(parameters);

        if (loss.Shape.Rank != 0)
        {
            throw new ArgumentException(
                $"A gradient is taken of one value, a loss with no axes, and this loss is {loss.Shape}.", nameof(loss));
        }

        return Worked(loss, parameters, _inner.Fill(new Shape(), 1f));
    }

    /// <summary>How much each of the given tensors moved a result this pass worked out, when each value of it is worth the share a seed says.</summary>
    /// <param name="loss">The result: any shape, worked out in this pass.</param>
    /// <param name="parameters">The tensors whose gradients are wanted — each one the pass read.</param>
    /// <param name="seed">How much each value of the result is worth: shaped as the result is. A one-valued loss with a seed of two to the sixteenth is the loss scaled by it, and every gradient comes back scaled by it exactly.</param>
    /// <returns>A gradient for each, of its own shape: the way back from the seed, so the sum over the result's values of the seed times how much each of them moved the tensor.</returns>
    /// <exception cref="ArgumentException">The seed is not shaped as the result is, the result was not worked out in this pass, or a tensor asked for is one the pass never read.</exception>
    /// <remarks>
    /// The loss's own gradient is one, and this is the same pass with another number there: a seed of one is the loss, a seed of a power
    /// of two is the loss scaled so that small gradients survive a format with little room, and a seed that is not a one-valued loss's is
    /// the vector-Jacobian product a part of a pass is worked back through when the rest of it has been worked back to its result.
    /// </remarks>
    public Gradients GradientsOf(Tensor loss, IEnumerable<Tensor> parameters, Tensor seed)
    {
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(seed);

        if (seed.Shape != loss.Shape)
        {
            throw new ArgumentException(
                $"A seed says how much each value of the result is worth, so it is shaped as the result is: this result is {loss.Shape} and this seed {seed.Shape}.", nameof(seed));
        }

        return Worked(loss, parameters, seed);
    }

    private Gradients Worked(Tensor loss, IEnumerable<Tensor> parameters, Tensor seed)
    {
        IReadOnlyList<Tensor> asked = [.. parameters];

        var last = _steps.FindLastIndex(step => ReferenceEquals(step.Output, loss));

        if (last < 0)
        {
            throw new ArgumentException(
                "The loss was not worked out in this pass, so there is nothing to work back through.", nameof(loss));
        }

        var read = new HashSet<Tensor>(_steps.Take(last + 1).SelectMany(step => step.Inputs), ReferenceEqualityComparer.Instance);

        if (asked.FirstOrDefault(parameter => !read.Contains(parameter)) is { } unread)
        {
            throw new ArgumentException(
                $"A {unread.Shape} tensor was asked for that this pass never read before its loss, so the loss cannot have moved with it.",
                nameof(parameters));
        }

        var towards = Towards(asked, last);
        var widens = _inner as IWidens;
        var asks = new HashSet<Tensor>(asked, ReferenceEqualityComparer.Instance);
        var received = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance) { [loss] = seed };

        for (var at = last; at >= 0; at--)
        {
            var step = _steps[at];

            if (!received.TryGetValue(step.Output, out var gradient))
            {
                continue;
            }

            for (var input = 0; input < step.Inputs.Count; input++)
            {
                var to = step.Inputs[input];

                // A share that could reach no tensor asked for is never worked out.
                if (towards.Contains(to))
                {
                    var share = step.Back(_inner, gradient, input);

                    // A number a network learns gathers the shares sent back to it at full width, as an autograd leaf behind a cast does.
                    if (widens is not null && asks.Contains(to))
                    {
                        share = widens.Widened(share);
                    }

                    received[to] = received.TryGetValue(to, out var sofar) ? _inner.Add(sofar, share) : share;
                }
            }
        }

        var gradients = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance);

        foreach (var parameter in asked)
        {
            gradients[parameter] = received.TryGetValue(parameter, out var gradient) ? gradient : _inner.Fill(parameter.Shape, 0f);
        }

        return new Gradients(gradients);
    }

    // Every tensor a share is worked out for on the way back: the ones asked for, and whatever the pass made of one of them
    // before the loss — an operation leads back to a tensor asked for when one of the tensors it read does.
    private HashSet<Tensor> Towards(IReadOnlyList<Tensor> asked, int last)
    {
        var towards = new HashSet<Tensor>(asked, ReferenceEqualityComparer.Instance);

        for (var at = 0; at <= last; at++)
        {
            var step = _steps[at];

            for (var input = 0; input < step.Inputs.Count; input++)
            {
                if (towards.Contains(step.Inputs[input]))
                {
                    towards.Add(step.Output);

                    break;
                }
            }
        }

        return towards;
    }

    /// <summary>
    /// Runs a layer over what its steps made on the backend this records, writing nothing of it down, and writes down one step in its
    /// place that works the layer out again, and only then, when the way back reaches it.
    /// </summary>
    /// <param name="layer">The layer.</param>
    /// <param name="inputs">What the steps it takes made.</param>
    /// <param name="pass">The pass it is run in.</param>
    /// <returns>What the layer makes.</returns>
    internal Tensor Recomputed(Layer layer, IReadOnlyList<Tensor> inputs, Pass pass)
    {
        var made = layer.Taking(inputs, pass.On(_inner));

        // A layer that hands back what it was handed has no workings to throw away, and a step whose output is its own input is no step.
        if (inputs.Any(input => ReferenceEquals(input, made)))
        {
            return made;
        }

        IReadOnlyList<Tensor> read = [.. inputs.Concat(layer.Parameters().Select(parameter => parameter.Value)).Distinct(ReferenceEqualityComparer.Instance as IEqualityComparer<Tensor>)];

        return Kept(new Recomputation(layer, pass, inputs, read, made));
    }

    // Whether this pass read a tensor in any operation it wrote down.
    private bool Reads(Tensor tensor) => _steps.Any(step => step.Inputs.Any(input => ReferenceEquals(input, tensor)));

    private Tensor Kept(Recorded step)
    {
        _steps.Add(step);

        return step.Output;
    }

    // One operation of a pass: what it made, what it read — in order, a tensor read twice named twice — and the rule that
    // sends a gradient back through it to what it read at one place.
    private abstract record Recorded(Tensor Output, IReadOnlyList<Tensor> Inputs)
    {
        public abstract Tensor Back(ITensorBackend backend, Tensor gradient, int input);
    }

    private sealed record Added(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output, [Left, Right])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => gradient;
    }

    private sealed record Subtracted(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output, [Left, Right])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            input == 0 ? gradient : backend.Scale(gradient, backend.Fill(new Shape(), -1f));
    }

    private sealed record Multiplied(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output, [Left, Right])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Multiply(gradient, input == 0 ? Right : Left);
    }

    private sealed record MatrixProduct(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output, [Left, Right])
    {
        // An engine that has the product over blocks and not the turned product of its own multiplies a matrix turned on its side as a single block
        // with the matrix laid on its side, and makes no turned copy: a copy of a matrix that is not in single precision is a slow thing to make.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend is IMultipliesInBlocks blocks and not IMultipliesTurnedLeft
                ? input == 0 ? blocks.MatMulInBlocks(gradient, Right, new BlockProduct(1) { TurnRight = true }) : blocks.MatMulInBlocks(Left, gradient, new BlockProduct(1) { TurnLeft = true })
                : input == 0 ? backend.MatMul(gradient, backend.Transpose(Right))
                : backend is IMultipliesTurnedLeft turning ? turning.MatMulTurnedLeft(Left, gradient)
                : backend.MatMul(backend.Transpose(Left), gradient);
    }

    // The gradient of block i of the first is the answer's gradient times block i of the second turned the other way, and of the second the first
    // turned the other way times the answer's gradient: a product of blocks again, asked of the wrapped engine as it was.
    private sealed record BlockedProduct(Tensor Left, Tensor Right, BlockProduct How, Tensor Output) : Recorded(Output, [Left, Right])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => input == 0 ? ToLeft(backend, gradient) : ToRight(backend, gradient);

        private Tensor ToLeft(ITensorBackend backend, Tensor gradient) =>
            How.TurnLeft
                ? backend.MatMulInBlocks(Right, gradient, new BlockProduct(How.Count) { TurnLeft = How.TurnRight, TurnRight = true })
                : backend.MatMulInBlocks(gradient, Right, new BlockProduct(How.Count) { TurnRight = !How.TurnRight });

        private Tensor ToRight(ITensorBackend backend, Tensor gradient) =>
            How.TurnRight
                ? backend.MatMulInBlocks(gradient, Left, new BlockProduct(How.Count) { TurnLeft = true, TurnRight = How.TurnLeft })
                : backend.MatMulInBlocks(Left, gradient, new BlockProduct(How.Count) { TurnLeft = !How.TurnLeft });
    }

    private sealed record Transposed(Tensor Matrix, Tensor Output) : Recorded(Output, [Matrix])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => backend.Transpose(gradient);
    }

    private sealed record RowAdded(Tensor Matrix, Tensor Row, Tensor Output) : Recorded(Output, [Matrix, Row])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            input == 0 ? gradient : backend.SumRows(gradient);
    }

    private sealed record RowsSummed(Tensor Matrix, Tensor Output) : Recorded(Output, [Matrix])
    {
        // Every row added into the total, so every row is sent the total's gradient.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.AddRow(backend.Fill(Matrix.Shape, 0f), gradient);
    }

    private sealed record Averaged(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        // A mean refuses no values, so a share is always one of so many.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Scale(backend.Fill(Values.Shape, 1f / Values.Shape.Count), gradient);
    }

    private sealed record Scaled(Tensor Values, Tensor Factor, Tensor Output) : Recorded(Output, [Values, Factor])
    {
        // The factor moved every value it scaled, so it is sent the total of the gradient times each value: the mean
        // taken so many times, and nothing when it scaled nothing.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input)
        {
            if (input == 0)
            {
                return backend.Scale(gradient, Factor);
            }

            return Values.Shape.Count == 0
                ? backend.Fill(new Shape(), 0f)
                : backend.Scale(backend.Mean(backend.Multiply(gradient, Values)), backend.Fill(new Shape(), Values.Shape.Count));
        }
    }

    private sealed record Rectified(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        // The gradient passes where the value was above nothing, as PyTorch lets it: nothing at nothing itself.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Multiply(gradient, backend.Positive(Values));
    }

    private sealed record Stepped(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        // A step is flat either side of where it steps, so it moves nothing it read.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => backend.Fill(Values.Shape, 0f);
    }

    private sealed record TanhOf(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Multiply(gradient, backend.Subtract(backend.Fill(Output.Shape, 1f), backend.Multiply(Output, Output)));
    }

    private sealed record SigmoidOf(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Multiply(gradient, backend.Multiply(Output, backend.Subtract(backend.Fill(Output.Shape, 1f), Output)));
    }

    private sealed record Exponentiated(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => backend.Multiply(gradient, Output);
    }

    private sealed record LogarithmOf(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => backend.Divide(gradient, Values);
    }

    private sealed record SquareRootOf(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Divide(gradient, backend.Add(Output, Output));
    }

    private sealed record SoftplusOf(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Multiply(gradient, backend.Sigmoid(Values));
    }

    // The sine's slope is its cosine, which is the sine a quarter of a turn on.
    private sealed record SineOf(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Multiply(gradient, backend.Sin(backend.Add(Values, backend.Fill(Values.Shape, (float)(Math.PI / 2)))));
    }

    // The error function's slope is two over the root of π times the exponential of minus the square.
    private sealed record ErrorFunctionOf(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            backend.Multiply(
                gradient,
                backend.Scale(backend.Exp(backend.Scale(backend.Multiply(Values, Values), backend.Fill(new Shape(), -1f))), backend.Fill(new Shape(), (float)(2 / Math.Sqrt(Math.PI)))));
    }

    private sealed record Divided(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output, [Left, Right])
    {
        // The bottom is sent minus the quotient over itself, times the gradient.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) =>
            input == 0
                ? backend.Divide(gradient, Right)
                : backend.Scale(backend.Divide(backend.Multiply(gradient, Output), Right), backend.Fill(new Shape(), -1f));
    }

    private sealed record LogSharesOf(Tensor Matrix, Tensor Output) : Recorded(Output, [Matrix])
    {
        // Every value of a row moved the row's shift, so each is sent its gradient less its share of the row's whole
        // gradient: the row's total, laid across the row by turning the matrix round and adding it as a row.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input)
        {
            var totals = backend.SumRows(backend.Transpose(gradient));
            var across = backend.Transpose(backend.AddRow(backend.Fill(new Shape(Matrix.Shape[1], Matrix.Shape[0]), 0f), totals));

            return backend.Subtract(gradient, backend.Multiply(backend.Exp(Output), across));
        }
    }

    private sealed record Reshaped(Tensor Values, Tensor Output) : Recorded(Output, [Values])
    {
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => backend.Reshape(gradient, Values.Shape);
    }

    private sealed record Unfolded(Tensor Images, Window Window, Tensor Output) : Recorded(Output, [Images])
    {
        // Every patch value came from one place of the images, so folding the gradient back adds it there.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => backend.Fold(gradient, Images.Shape, Window);
    }

    // A layer thrown away after its forward pass: when the way back reaches it, it is run again on a recording of its own, from the
    // same draws and moving no statistic, and the gradient it was handed is worked back through that — once, whichever of the things it
    // read is asked for first.
    private sealed record Recomputation(Layer Layer, Pass Pass, IReadOnlyList<Tensor> Taken, IReadOnlyList<Tensor> Read, Tensor Output) : Recorded(Output, Read)
    {
        private Gradients? _worked;
        private HashSet<Tensor>? _reached;

        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input)
        {
            if (_worked is null)
            {
                var again = new RecordingBackend(backend);
                var remade = Layer.Taking(Taken, Pass.Replaying(again));
                var reached = Read.Where(again.Reads).ToArray();

                _reached = new HashSet<Tensor>(reached, ReferenceEqualityComparer.Instance);
                _worked = again.GradientsOf(remade, reached, gradient);
            }

            return _reached!.Contains(Read[input]) ? _worked[Read[input]] : backend.Fill(Read[input].Shape, 0f);
        }
    }

    private sealed record Folded(Tensor Patches, Window Window, Tensor Output) : Recorded(Output, [Patches])
    {
        // Every patch value was added into one place, so each is sent that place's gradient: the gradient unfolded.
        public override Tensor Back(ITensorBackend backend, Tensor gradient, int input) => backend.Unfold(gradient, Window);
    }
}

/// <summary>How much each tensor asked for moved a pass's loss: one gradient each, of that tensor's own shape.</summary>
/// <remarks>Each tensor is known by which one it is, never by the values it holds, so two tensors that happen to hold the same numbers keep their own gradients.</remarks>
public sealed class Gradients
{
    private readonly Dictionary<Tensor, Tensor> _of;

    internal Gradients(Dictionary<Tensor, Tensor> of) => _of = of;

    /// <summary>These gradients and others of the same tensors, added up tensor by tensor.</summary>
    /// <param name="other">The others: one for each tensor these hold.</param>
    /// <param name="backend">The engine that adds.</param>
    /// <returns>The sums.</returns>
    internal Gradients Added(Gradients other, ITensorBackend backend) =>
        new(_of.ToDictionary(each => each.Key, each => backend.Add(each.Value, other[each.Key])));

    /// <summary>The tensors these gradients are of, each once.</summary>
    internal IEnumerable<Tensor> Tensors => _of.Keys;

    /// <summary>These gradients, each multiplied by one number.</summary>
    /// <param name="factor">The number: a power of two when the gradients are scaled down by the scale they were worked back from.</param>
    /// <param name="backend">The engine that scales.</param>
    /// <returns>The scaled gradients.</returns>
    internal Gradients Scaled(double factor, ITensorBackend backend)
    {
        var by = backend.Fill(new Shape(), (float)factor);

        return new(_of.ToDictionary(each => each.Key, each => backend.Scale(each.Value, by)));
    }

    /// <summary>The sum of the squares of every value of every gradient: not a number if any value of any of them is not a finite number.</summary>
    /// <param name="backend">The engine that works it out.</param>
    /// <returns>The sum, each tensor's mean square counted back up by the values it holds, and added in double.</returns>
    internal double Squares(ITensorBackend backend)
    {
        var squares = 0d;

        foreach (var gradient in _of.Values.Where(tensor => tensor.Shape.Count > 0))
        {
            squares += (double)backend.Mean(backend.Multiply(gradient, gradient)).Values[0] * gradient.Shape.Count;
        }

        return squares;
    }

    /// <summary>These gradients, each divided by a count: the mean of that many added up.</summary>
    /// <param name="count">How many were added up.</param>
    /// <param name="backend">The engine that scales.</param>
    /// <returns>The means; these as they stand, for a count of one.</returns>
    internal Gradients Averaged(int count, ITensorBackend backend) =>
        count == 1
            ? this
            : new(_of.ToDictionary(each => each.Key, each => backend.Scale(each.Value, backend.Fill(new Shape(), (float)(1.0 / count)))));

    /// <summary>These gradients with every value held between minus a bound and the bound: PyTorch's <c>clip_grad_value_</c>.</summary>
    /// <param name="parameters">The parameters the gradients are of, whose names a refusal uses.</param>
    /// <param name="bound">How far either side of nothing a value may reach.</param>
    /// <param name="backend">The engine that clips.</param>
    /// <returns>The clipped gradients.</returns>
    /// <exception cref="InvalidOperationException">A gradient holds a value that is not a finite number, which a bound says nothing of.</exception>
    internal Gradients Clamped(IReadOnlyList<Parameter> parameters, double bound, ITensorBackend backend)
    {
        var clipped = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance);

        foreach (var parameter in parameters)
        {
            var gradient = this[parameter.Value];

            if (!gradient.Values.ToArray().All(float.IsFinite))
            {
                throw new InvalidOperationException(
                    $"The gradient of '{parameter.Name}' holds a value that is not a finite number, and clipping each value to a bound says nothing of one: the rows, the rate or the network's start have driven it past what a float holds.");
            }

            clipped[parameter.Value] = backend.Clamped(gradient, (float)bound);
        }

        return new Gradients(clipped);
    }

    /// <summary>The gradient of one of the tensors asked for.</summary>
    /// <param name="parameter">The tensor, as it was handed to the pass.</param>
    /// <returns>Its gradient, of its own shape.</returns>
    /// <exception cref="ArgumentException">The tensor was not among those asked for.</exception>
    public Tensor this[Tensor parameter]
    {
        get
        {
            ArgumentNullException.ThrowIfNull(parameter);

            return _of.TryGetValue(parameter, out var gradient)
                ? gradient
                : throw new ArgumentException($"A gradient was not asked for this {parameter.Shape} tensor.", nameof(parameter));
        }
    }
}
