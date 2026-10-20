// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>What a recurrent layer reads and how it is built: the width of a step, the width of its state, its layers, its directions.</summary>
/// <param name="Inputs">How many values each step of a series holds: the channels of a series.</param>
/// <param name="Hidden">How many numbers the state the layer carries along a series holds.</param>
public readonly record struct Recurrence(int Inputs, int Hidden)
{
    /// <summary>How many layers are stacked, each reading the steps the one below it made; one, unless said.</summary>
    public int Layers { get; init; } = 1;

    /// <summary>Whether a second state reads each series from its own end back, and the two are laid side by side; it does not, unless said.</summary>
    public bool BothWays { get; init; }

    /// <summary>Whether the layer makes what its state was at every step, rather than at the last; it makes the last, unless said.</summary>
    public bool Sequences { get; init; }
}

/// <summary>
/// A layer that reads a series a step at a time and carries a state along it, as PyTorch's <c>nn.LSTM</c> and <c>nn.GRU</c> do over a batch
/// of series, steps and then channels: one layer or several stacked, in one direction or in both.
/// </summary>
/// <remarks>
/// <para>
/// A layer holds, for each of its layers and each direction, four numbers named as PyTorch names them — <c>weight_ih_l0</c>,
/// <c>weight_hh_l0</c>, <c>bias_ih_l0</c>, <c>bias_hh_l0</c>, and the same with <c>_reverse</c> for the direction back — the gates one block
/// after another along the outputs, in PyTorch's order. The weights are kept inputs by outputs, as a linear layer's are, so a reader turns the
/// ones PyTorch writes outputs by inputs.
/// </para>
/// <para>
/// A batch of series of unequal lengths is read each as it would be read alone: given a mask of the steps that are steps of a series (one) and
/// those that are padding (nought), a padding step leaves the state as it was and makes nothing, and a series read backwards starts from its
/// own last step. It makes the last state at the series' own length, or every step with nothing at the padding, as PyTorch makes a packed batch.
/// </para>
/// <para>
/// Every step is made of the products the seam already has, so the gradient of a loss through a series is the one the tape works out through
/// the steps, back to the first, and no engine learned anything for it.
/// </para>
/// </remarks>
public abstract class Recurrent : Layer
{
    private const string InputsKey = "inputs";
    private const string HiddenKey = "hidden";
    private const string LayersKey = "layers";
    private const string BothWaysKey = "bothWays";
    private const string SequencesKey = "sequences";

    private readonly Parameter[] _weightsIn;
    private readonly Parameter[] _weightsHidden;
    private readonly Parameter[] _biasesIn;
    private readonly Parameter[] _biasesHidden;

    private protected Recurrent(Recurrence recurrence, int gates, Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);
        ArgumentOutOfRangeException.ThrowIfLessThan(recurrence.Inputs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(recurrence.Hidden, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(recurrence.Layers, 1);

        Recurrence = recurrence;
        Gates = gates;

        var each = Directions * recurrence.Layers;
        var bound = 1 / Math.Sqrt(recurrence.Hidden);
        var wide = gates * recurrence.Hidden;

        _weightsIn = new Parameter[each];
        _weightsHidden = new Parameter[each];
        _biasesIn = new Parameter[each];
        _biasesHidden = new Parameter[each];

        for (var layer = 0; layer < recurrence.Layers; layer++)
        {
            for (var direction = 0; direction < Directions; direction++)
            {
                var at = (layer * Directions) + direction;
                var suffix = string.Create(CultureInfo.InvariantCulture, $"_l{layer}{(direction == 1 ? "_reverse" : string.Empty)}");
                var reads = layer == 0 ? recurrence.Inputs : recurrence.Hidden * Directions;

                _weightsIn[at] = AddParameter($"weight_ih{suffix}", Drawn(new Shape(reads, wide), draws, bound));
                _weightsHidden[at] = AddParameter($"weight_hh{suffix}", Drawn(new Shape(recurrence.Hidden, wide), draws, bound));
                _biasesIn[at] = AddParameter($"bias_ih{suffix}", Drawn(new Shape(wide), draws, bound));
                _biasesHidden[at] = AddParameter($"bias_hh{suffix}", Drawn(new Shape(wide), draws, bound));
            }
        }
    }

    /// <summary>What the layer reads and how it is built.</summary>
    public Recurrence Recurrence { get; }

    /// <summary>How many directions each layer reads in: one, or two where it reads each series back from its end as well.</summary>
    public int Directions => Recurrence.BothWays ? 2 : 1;

    /// <summary>How many gates a step is made of: four for an LSTM, three for a GRU.</summary>
    public int Gates { get; }

    /// <summary>Runs the layer over a batch of series of unequal lengths.</summary>
    /// <param name="input">The series: a batch of series by steps by channels.</param>
    /// <param name="mask">Which steps are steps of a series, one, and which are padding, nought: a batch by steps.</param>
    /// <param name="pass">The backend the arithmetic runs on, and whether this pass trains.</param>
    /// <returns>The last state of each series at its own length, or every step with nothing at the padding.</returns>
    /// <exception cref="ArgumentException">The input is not a batch of series of as many channels as the layer reads, or the mask is of another shape.</exception>
    public Tensor Forward(Tensor input, Tensor mask, Pass pass)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(mask);
        ArgumentNullException.ThrowIfNull(pass);

        return Run(input, mask, pass);
    }

    /// <inheritdoc />
    internal override bool Accepts(int steps) => steps is 1 or 2;

    /// <inheritdoc />
    internal override Tensor Taking(IReadOnlyList<Tensor> inputs, Pass pass) => Run(inputs[0], inputs.Count == 2 ? inputs[1] : null, pass);

    /// <inheritdoc />
    protected sealed override Tensor Compute(Tensor input, Pass pass) => Run(input, null, pass);

    /// <summary>The step: the state after one step, from the gates' projections of the step and of the state before it.</summary>
    /// <param name="backend">The engine the arithmetic runs on.</param>
    /// <param name="projections">The projections of the step's channels and of the state, each with its bias, the gates one after another.</param>
    /// <param name="before">The state before the step.</param>
    /// <returns>The state after the step.</returns>
    private protected abstract State Stepped(ITensorBackend backend, Projections projections, State before);

    /// <summary>The state a series starts from: nothing.</summary>
    /// <param name="backend">The engine the arithmetic runs on.</param>
    /// <param name="shape">One state: the hidden numbers by the series.</param>
    private protected abstract State Begun(ITensorBackend backend, Shape shape);

    /// <summary>The numbers of one gate of a projection: the rows of the gate's block.</summary>
    /// <param name="backend">The engine the arithmetic runs on.</param>
    /// <param name="projection">The projection: the gates' blocks one after another, by the series.</param>
    /// <param name="gate">Which gate, from nought.</param>
    private protected Tensor Gate(ITensorBackend backend, Tensor projection, int gate)
    {
        var hidden = Recurrence.Hidden;
        var series = projection.Shape[1];
        var blocks = backend.Reshape(projection, new Shape(Gates, hidden * series));

        return backend.Reshape(backend.MatMul(Pick(Gates, gate), blocks), new Shape(hidden, series));
    }

    /// <summary>The build a layer was written with, read back.</summary>
    /// <param name="settings">The object the layer was written as.</param>
    /// <param name="rebuilding">The reading of the file.</param>
    private protected static Recurrence Built(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new Recurrence(rebuilding.Whole(settings, InputsKey), rebuilding.Whole(settings, HiddenKey))
        {
            Layers = rebuilding.Whole(settings, LayersKey, 1),
            BothWays = rebuilding.Holds(settings, BothWaysKey),
            Sequences = rebuilding.Holds(settings, SequencesKey),
        };
    }

    /// <summary>Writes the build: what is read, how wide the state is and how many layers, and the two that say yes only where they do.</summary>
    /// <param name="writer">The writer.</param>
    private protected void WriteBuild(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(InputsKey, Recurrence.Inputs);
        writer.WriteNumber(HiddenKey, Recurrence.Hidden);
        writer.WriteNumber(LayersKey, Recurrence.Layers);

        if (Recurrence.BothWays)
        {
            writer.WriteBoolean(BothWaysKey, true);
        }

        if (Recurrence.Sequences)
        {
            writer.WriteBoolean(SequencesKey, true);
        }
    }

    private static Tensor Drawn(Shape shape, Draws draws, double bound) =>
        Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(_ => (float)(bound * ((2 * draws.NextDouble()) - 1)))]);

    // The row of a matrix of so many rows that holds the one at a place: a one at that place, noughts elsewhere.
    private static Tensor Pick(int count, int at) => Tensor.From(new Shape(1, count), [.. Enumerable.Range(0, count).Select(place => place == at ? 1f : 0f)]);

    private Tensor Run(Tensor input, Tensor? mask, Pass pass)
    {
        RequireSeries(input, mask);

        var backend = pass.Backend;
        var markers = mask is null ? null : backend.Transpose(mask);
        var current = input;

        for (var layer = 0; layer < Recurrence.Layers; layer++)
        {
            var top = layer == Recurrence.Layers - 1;
            var collect = !top || Recurrence.Sequences;
            var forward = Walk(backend, current, layer * Directions, new Direction(Reverse: false, collect), markers);
            var backward = Recurrence.BothWays ? Walk(backend, current, (layer * Directions) + 1, new Direction(Reverse: true, collect), markers) : default;

            if (top && !Recurrence.Sequences)
            {
                return Recurrence.BothWays ? Beside(backend, forward.Last, backward.Last) : forward.Last;
            }

            current = Recurrence.BothWays ? Beside(backend, forward.Sequence!, backward.Sequence!) : forward.Sequence!;
        }

        return current;
    }

    // One direction of one layer over a batch of series: the state carried along them, a step at a time, in the layout of the hidden numbers
    // by the series, so a gate is a block of rows and the products are weights times states.
    private Walked Walk(ITensorBackend backend, Tensor series, int at, Direction direction, Tensor? markers)
    {
        var count = series.Shape[0];
        var steps = series.Shape[1];
        var reads = series.Shape[2];
        var hidden = Recurrence.Hidden;
        var weightsIn = backend.Transpose(_weightsIn[at].Value);
        var weightsHidden = backend.Transpose(_weightsHidden[at].Value);
        var biasIn = Alongside(backend, _biasesIn[at].Value, count);
        var biasHidden = Alongside(backend, _biasesHidden[at].Value, count);
        var timeMajor = backend.Reshape(backend.Transpose(backend.Reshape(series, new Shape(count, steps * reads))), new Shape(steps, reads * count));
        var state = Begun(backend, new Shape(hidden, count));
        var gathered = direction.Collect ? backend.Fill(new Shape(steps, hidden * count), 0f) : null;

        for (var step = 0; step < steps; step++)
        {
            var place = direction.Reverse ? steps - 1 - step : step;
            var channels = backend.Reshape(backend.MatMul(Pick(steps, place), timeMajor), new Shape(reads, count));
            var projections = new Projections(
                backend.Add(backend.MatMul(weightsIn, channels), biasIn),
                backend.Add(backend.MatMul(weightsHidden, state.Hidden), biasHidden));
            var made = Stepped(backend, projections, state);
            var shown = made.Hidden;

            if (markers is not null)
            {
                var kept = backend.MatMul(backend.Fill(new Shape(hidden, 1), 1f), backend.MatMul(Pick(steps, place), markers));

                shown = backend.Multiply(kept, made.Hidden);
                made = new State(Held(backend, made.Hidden, state.Hidden, kept), made.Cell is null ? null : Held(backend, made.Cell, state.Cell!, kept));
            }

            state = made;

            if (gathered is not null)
            {
                var placed = backend.MatMul(Tensor.From(new Shape(steps, 1), [.. Enumerable.Range(0, steps).Select(row => row == place ? 1f : 0f)]), backend.Reshape(shown, new Shape(1, hidden * count)));

                gathered = backend.Add(gathered, placed);
            }
        }

        var sequence = gathered is null ? null : backend.Reshape(backend.Transpose(backend.Reshape(gathered, new Shape(steps * hidden, count))), new Shape(count, steps, hidden));

        return new Walked(sequence, backend.Transpose(state.Hidden));
    }

    // What a step that is a step of the series makes, and what a padding step leaves as it was: the new state where the mask says one.
    private static Tensor Held(ITensorBackend backend, Tensor made, Tensor before, Tensor kept) =>
        backend.Add(before, backend.Multiply(kept, backend.Subtract(made, before)));

    // A bias beside each series: the same numbers in every column of the layout the products are made in.
    private static Tensor Alongside(ITensorBackend backend, Tensor bias, int count) =>
        backend.MatMul(backend.Reshape(bias, new Shape(bias.Shape[0], 1)), backend.Fill(new Shape(1, count), 1f));

    // The two sides of a series laid side by side along the last axis: the forward state's numbers, then the backward one's.
    private static Tensor Beside(ITensorBackend backend, Tensor forward, Tensor backward)
    {
        var leading = forward.Shape.Axes[..^1].ToArray();
        var width = forward.Shape[forward.Shape.Rank - 1];
        var rows = leading.Aggregate(1, (all, length) => all * length);
        var first = backend.MatMul(backend.Reshape(forward, new Shape(rows, width)), Selection.Of(width, 2 * width, (place, column) => column == place));
        var second = backend.MatMul(backend.Reshape(backward, new Shape(rows, width)), Selection.Of(width, 2 * width, (place, column) => column == width + place));

        return backend.Reshape(backend.Add(first, second), new Shape([.. leading, 2 * width]));
    }

    private void RequireSeries(Tensor input, Tensor? mask)
    {
        if (input.Shape.Rank != 3 || input.Shape[2] != Recurrence.Inputs || input.Shape[1] < 1)
        {
            throw new ArgumentException(
                $"A recurrent layer of {Recurrence.Inputs} channels reads a batch of series, series by step by channel, of one step at least, and was handed a {input.Shape} one.",
                nameof(input));
        }

        if (mask is not null && mask.Shape != new Shape(input.Shape[0], input.Shape[1]))
        {
            throw new ArgumentException(
                $"The mask of a batch of {input.Shape} series is a one or a nought for each step of each, series by step, and was handed a {mask.Shape} one.",
                nameof(mask));
        }
    }

    /// <summary>The gates' projections of a step: of its channels, and of the state before it, each with its bias.</summary>
    /// <param name="FromInput">The channels of the step through the input weights.</param>
    /// <param name="FromHidden">The state before the step through the hidden weights.</param>
    private protected readonly record struct Projections(Tensor FromInput, Tensor FromHidden);

    /// <summary>The state carried along a series: the numbers a step reads and makes, and, for an LSTM, the cell beside them.</summary>
    /// <param name="Hidden">The hidden numbers, by the series.</param>
    /// <param name="Cell">The cell state, by the series; nothing for a layer that has none.</param>
    private protected readonly record struct State(Tensor Hidden, Tensor? Cell);

    private readonly record struct Direction(bool Reverse, bool Collect);

    private readonly record struct Walked(Tensor? Sequence, Tensor Last);
}

/// <summary>
/// A long short-term memory layer, as PyTorch's <c>nn.LSTM</c>: a cell state beside the hidden one, four gates — input, forget, cell, output —
/// the cell kept by the forget gate and what the input gate lets in, and the hidden numbers the output gate's share of its curve.
/// </summary>
public sealed class Lstm : Recurrent, ISaved<Lstm>
{
    /// <summary>An LSTM of the given build, its numbers starting from the given draws as PyTorch starts them: evenly within one over the root of the hidden numbers.</summary>
    /// <param name="recurrence">What it reads and how it is built.</param>
    /// <param name="draws">The draws its start is taken from.</param>
    /// <exception cref="ArgumentOutOfRangeException">A step, a state or the layers hold fewer than one number.</exception>
    /// <exception cref="ArgumentNullException">There are no draws.</exception>
    public Lstm(Recurrence recurrence, Draws draws)
        : base(recurrence, 4, draws)
    {
    }

    /// <inheritdoc />
    public static string Name => "lstm";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static Lstm Rebuild(JsonElement settings, Rebuilding rebuilding) => new(Built(settings, rebuilding), rebuilding.Draws);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer) => WriteBuild(writer);

    /// <inheritdoc />
    private protected override State Begun(ITensorBackend backend, Shape shape) => new(backend.Fill(shape, 0f), backend.Fill(shape, 0f));

    /// <inheritdoc />
    private protected override State Stepped(ITensorBackend backend, Projections projections, State before)
    {
        var gates = backend.Add(projections.FromInput, projections.FromHidden);
        var input = backend.Sigmoid(Gate(backend, gates, 0));
        var forget = backend.Sigmoid(Gate(backend, gates, 1));
        var cell = backend.Tanh(Gate(backend, gates, 2));
        var output = backend.Sigmoid(Gate(backend, gates, 3));
        var kept = backend.Add(backend.Multiply(forget, before.Cell!), backend.Multiply(input, cell));

        return new State(backend.Multiply(output, backend.Tanh(kept)), kept);
    }
}

/// <summary>
/// A gated recurrent unit, as PyTorch's <c>nn.GRU</c>: three gates — reset, update, new — the reset gate taken of what the hidden numbers make
/// after their own bias, and the hidden numbers a share of the new ones and of the old, the update gate's.
/// </summary>
public sealed class Gru : Recurrent, ISaved<Gru>
{
    /// <summary>A GRU of the given build, its numbers starting from the given draws as PyTorch starts them: evenly within one over the root of the hidden numbers.</summary>
    /// <param name="recurrence">What it reads and how it is built.</param>
    /// <param name="draws">The draws its start is taken from.</param>
    /// <exception cref="ArgumentOutOfRangeException">A step, a state or the layers hold fewer than one number.</exception>
    /// <exception cref="ArgumentNullException">There are no draws.</exception>
    public Gru(Recurrence recurrence, Draws draws)
        : base(recurrence, 3, draws)
    {
    }

    /// <inheritdoc />
    public static string Name => "gru";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static Gru Rebuild(JsonElement settings, Rebuilding rebuilding) => new(Built(settings, rebuilding), rebuilding.Draws);

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer) => WriteBuild(writer);

    /// <inheritdoc />
    private protected override State Begun(ITensorBackend backend, Shape shape) => new(backend.Fill(shape, 0f), null);

    /// <inheritdoc />
    private protected override State Stepped(ITensorBackend backend, Projections projections, State before)
    {
        var reset = backend.Sigmoid(backend.Add(Gate(backend, projections.FromInput, 0), Gate(backend, projections.FromHidden, 0)));
        var update = backend.Sigmoid(backend.Add(Gate(backend, projections.FromInput, 1), Gate(backend, projections.FromHidden, 1)));
        var fresh = backend.Tanh(backend.Add(Gate(backend, projections.FromInput, 2), backend.Multiply(reset, Gate(backend, projections.FromHidden, 2))));

        return new State(backend.Add(fresh, backend.Multiply(update, backend.Subtract(before.Hidden, fresh))), null);
    }
}
