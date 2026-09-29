// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// What turns a gradient into a change: every parameter moved one step against its gradient.
/// </summary>
/// <remarks>
/// Its arithmetic runs on the backend it is handed, and never through a recording one — a step is no part of the pass
/// whose gradient it takes, so a recording backend handed in is looked through to the one it wraps. What it remembers
/// between steps — a velocity, a running mean of gradients — is kept for each parameter's slot, which stays the same
/// slot while its tensor is replaced. Every optimizer here works as PyTorch's of the same name, with its defaults; the
/// options PyTorch has beyond them — weight decay, Nesterov's momentum, AMSGrad — are left out, and each does nothing
/// there by default.
/// </remarks>
public abstract class Optimizer
{
    /// <summary>An optimizer that starts at the given rate.</summary>
    /// <param name="rate">How far the first epoch's steps move the parameters.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a number above nothing.</exception>
    protected Optimizer(double rate) => Rate = RequireRate(rate);

    /// <summary>How far a step moves the parameters, before a learning-rate schedule changes it from epoch to epoch.</summary>
    public double Rate { get; }

    /// <summary>Moves every parameter one step against its gradient.</summary>
    /// <param name="parameters">The parameters to move.</param>
    /// <param name="gradients">The gradients the pass worked out, one for each parameter's tensor as the pass read it.</param>
    /// <param name="rate">How far this step moves them: the rate the schedule gives this epoch.</param>
    /// <param name="backend">The backend the arithmetic runs on.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a number above nothing.</exception>
    /// <exception cref="ArgumentException">A parameter's gradient was not worked out.</exception>
    public void Step(IEnumerable<Parameter> parameters, Gradients gradients, double rate, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(gradients);
        ArgumentNullException.ThrowIfNull(backend);
        RequireRate(rate);

        var arithmetic = backend is RecordingBackend recording ? recording.Inner : backend;

        foreach (var parameter in parameters)
        {
            parameter.Replace(Moved(parameter, gradients[parameter.Value], rate, arithmetic));
        }
    }

    /// <summary>What the optimizer remembers of a parameter, to be kept in a checkpoint; nothing when it remembers nothing yet.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>The steps it took and the tensors kept for it, or nothing.</returns>
    internal abstract SlotMemory? MemoryOf(Parameter parameter);

    /// <summary>Puts back what a checkpoint says the optimizer remembered of a parameter.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <param name="memory">What was remembered of it.</param>
    /// <exception cref="ArgumentException">The memory is not one this optimizer keeps, or not of this parameter's shape.</exception>
    internal abstract void Recall(Parameter parameter, SlotMemory memory);

    /// <summary>A tensor of a remembered set, of the parameter's shape, or the refusal that says why not.</summary>
    private protected static Tensor Remembered(Parameter parameter, SlotMemory memory, string name) =>
        memory.Tensors.TryGetValue(name, out var tensor) && tensor.Shape == parameter.Value.Shape
            ? tensor
            : throw new ArgumentException(
                $"What the checkpoint remembers of '{parameter.Name}' has no {name} of its {parameter.Value.Shape} shape, which this optimizer keeps.",
                nameof(memory));

    /// <summary>Where one parameter moves to, from its tensor now and its gradient.</summary>
    /// <param name="parameter">The parameter, whose slot is what the optimizer remembers it by.</param>
    /// <param name="gradient">Its gradient.</param>
    /// <param name="rate">How far this step moves it.</param>
    /// <param name="backend">The backend the arithmetic runs on, which records nothing.</param>
    /// <returns>The tensor the parameter holds after the step.</returns>
    protected abstract Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend);

    /// <summary>One value with no axes: what a tensor is scaled by.</summary>
    /// <param name="backend">The backend.</param>
    /// <param name="value">The value.</param>
    /// <returns>The value as a tensor.</returns>
    private protected static Tensor Scalar(ITensorBackend backend, double value) => backend.Fill(new Shape(), (float)value);

    private static double RequireRate(double rate) =>
        double.IsFinite(rate) && rate > 0 ? rate : throw new ArgumentOutOfRangeException(nameof(rate), rate, "A learning rate is a number above nothing.");
}

/// <summary>
/// Stochastic gradient descent: every parameter moved by the rate times its gradient — or, with momentum, times a
/// velocity that carries on the gradients before it. PyTorch's <c>SGD</c>.
/// </summary>
public sealed class Sgd : Optimizer, ISaved<Sgd>
{
    private readonly Dictionary<Parameter, Tensor> _velocities = [];
    private readonly double _momentum;

    /// <summary>Stochastic gradient descent at the given rate.</summary>
    /// <param name="rate">How far a step moves the parameters; a thousandth, unless said, as PyTorch leaves it.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a number above nothing.</exception>
    public Sgd(double rate = 0.001)
        : base(rate)
    {
    }

    /// <summary>How much of the velocity carries on from one step to the next; none, unless said.</summary>
    /// <remarks>The first step's velocity is its gradient, and each after is the momentum times the last plus the gradient.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing or not a number.</exception>
    public double Momentum
    {
        get => _momentum;
        init => _momentum = double.IsFinite(value) && value >= 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Momentum is a number of at least nothing.");
    }

    /// <inheritdoc />
    public static string Name => "sgd";

    /// <inheritdoc />
    public static Sgd Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "rate")) { Momentum = rebuilding.Number(settings, "momentum") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("rate", Rate);
        writer.WriteNumber("momentum", Momentum);
    }

    /// <summary>The velocity a parameter carries into its next step.</summary>
    internal Tensor VelocityOf(Parameter parameter) => _velocities[parameter];

    /// <inheritdoc />
    internal override SlotMemory? MemoryOf(Parameter parameter) =>
        _velocities.TryGetValue(parameter, out var velocity)
            ? new SlotMemory(0, new Dictionary<string, Tensor> { ["momentum_buffer"] = velocity })
            : null;

    /// <inheritdoc />
    internal override void Recall(Parameter parameter, SlotMemory memory) =>
        _velocities[parameter] = Remembered(parameter, memory, "momentum_buffer");

    /// <inheritdoc />
    protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend)
    {
        var direction = gradient;

        if (Momentum > 0)
        {
            direction = _velocities.TryGetValue(parameter, out var velocity)
                ? backend.Add(backend.Scale(velocity, Scalar(backend, Momentum)), gradient)
                : gradient;

            _velocities[parameter] = direction;
        }

        return backend.Subtract(parameter.Value, backend.Scale(direction, Scalar(backend, rate)));
    }
}

/// <summary>The two rates Adam's running means of the gradients and of their squares forget at.</summary>
/// <param name="First">How much of the mean of the gradients carries on from one step to the next.</param>
/// <param name="Second">How much of the mean of their squares carries on.</param>
public readonly record struct Betas(double First, double Second);

/// <summary>
/// Adam: every parameter moved by a running mean of its gradients over the root of a running mean of their squares, both
/// corrected for starting at nothing. PyTorch's <c>Adam</c>, its arithmetic in PyTorch's order.
/// </summary>
public sealed class Adam : Optimizer, ISaved<Adam>
{
    private readonly Dictionary<Parameter, AdamMoments> _moments = [];
    private readonly Betas _betas = new(0.9, 0.999);
    private readonly double _epsilon = 1e-8;

    /// <summary>Adam at the given rate.</summary>
    /// <param name="rate">How far a step moves the parameters; a thousandth, unless said, as PyTorch leaves it.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a number above nothing.</exception>
    public Adam(double rate = 0.001)
        : base(rate)
    {
    }

    /// <summary>How much of each running mean carries on from one step to the next; nine tenths and 0.999, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either is not a share from nothing to below one.</exception>
    public Betas Betas
    {
        get => _betas;
        init => _betas = value.First is >= 0 and < 1 && value.Second is >= 0 and < 1
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Each beta is a share from nothing to below one.");
    }

    /// <summary>What is added to the root of the mean of the squares, so nothing is divided by nothing; a hundred-millionth, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number above nothing.</exception>
    public double Epsilon
    {
        get => _epsilon;
        init => _epsilon = double.IsFinite(value) && value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Epsilon is a number above nothing.");
    }

    /// <inheritdoc />
    public static string Name => "adam";

    /// <inheritdoc />
    public static Adam Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var betas = rebuilding.Numbers(settings, "betas");

        return betas.Count == 2
            ? new(rebuilding.Number(settings, "rate")) { Betas = new Betas(betas[0], betas[1]), Epsilon = rebuilding.Number(settings, "epsilon") }
            : throw new FormatException("'betas' is the two rates Adam's running means forget at, as a list of two numbers.");
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("rate", Rate);
        writer.WriteStartArray("betas");
        writer.WriteNumberValue(Betas.First);
        writer.WriteNumberValue(Betas.Second);
        writer.WriteEndArray();
        writer.WriteNumber("epsilon", Epsilon);
    }

    /// <summary>The steps a parameter has taken and its two running means.</summary>
    internal AdamMoments MomentsOf(Parameter parameter) => _moments[parameter];

    /// <inheritdoc />
    internal override SlotMemory? MemoryOf(Parameter parameter) =>
        _moments.TryGetValue(parameter, out var moments)
            ? new SlotMemory(moments.Steps, new Dictionary<string, Tensor> { ["exp_avg"] = moments.Average, ["exp_avg_sq"] = moments.SquaredAverage })
            : null;

    /// <inheritdoc />
    internal override void Recall(Parameter parameter, SlotMemory memory)
    {
        if (memory.Steps < 1)
        {
            throw new ArgumentException($"What the checkpoint remembers of '{parameter.Name}' counts no step, and Adam remembers only what stepped.", nameof(memory));
        }

        _moments[parameter] = new AdamMoments(memory.Steps, Remembered(parameter, memory, "exp_avg"), Remembered(parameter, memory, "exp_avg_sq"));
    }

    /// <inheritdoc />
    protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend)
    {
        var before = _moments.TryGetValue(parameter, out var kept)
            ? kept
            : new AdamMoments(0, backend.Fill(gradient.Shape, 0f), backend.Fill(gradient.Shape, 0f));

        var steps = before.Steps + 1;
        var average = backend.Add(
            backend.Scale(before.Average, Scalar(backend, Betas.First)), backend.Scale(gradient, Scalar(backend, 1 - Betas.First)));
        var squared = backend.Add(
            backend.Scale(before.SquaredAverage, Scalar(backend, Betas.Second)),
            backend.Scale(backend.Multiply(gradient, gradient), Scalar(backend, 1 - Betas.Second)));

        _moments[parameter] = new AdamMoments(steps, average, squared);

        // PyTorch's order: the root of the squares over the root of their correction, plus epsilon, divides the mean, and the
        // step is the rate over the mean's correction.
        var denominator = backend.Add(
            backend.Scale(backend.Sqrt(squared), Scalar(backend, 1 / Math.Sqrt(1 - Math.Pow(Betas.Second, steps)))),
            backend.Fill(gradient.Shape, (float)Epsilon));
        var size = rate / (1 - Math.Pow(Betas.First, steps));

        return backend.Subtract(parameter.Value, backend.Scale(backend.Divide(average, denominator), Scalar(backend, size)));
    }
}

/// <summary>What Adam remembers of one parameter: how many steps it took, and its running means of the gradients and of their squares.</summary>
internal readonly record struct AdamMoments(int Steps, Tensor Average, Tensor SquaredAverage);
