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
/// slot while its tensor is replaced. Every optimizer here works as PyTorch's of the same name, with its defaults. Of the
/// options PyTorch has beyond them, weight decay is built for <see cref="AdamW"/> alone, decoupled from the gradient as
/// AdamW decays; the weight decay PyTorch adds to the gradient of SGD, Adam, RMSprop and NAdam, Nesterov's momentum,
/// AMSGrad and RMSprop's centred form are not built, and each does nothing there by default.
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
    /// <remarks>
    /// What a run needs to go on as if it had never stopped, and nothing else: a velocity, a running mean, the number of steps
    /// taken. Each tensor is of the parameter's shape and is kept under the name PyTorch gives it, which is also the name a
    /// checkpoint's file writes it under. An optimizer that remembers nothing returns nothing, and puts back nothing.
    /// </remarks>
    protected abstract SlotMemory? MemoryOf(Parameter parameter);

    /// <summary>Puts back what a checkpoint says the optimizer remembered of a parameter.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <param name="memory">What was remembered of it.</param>
    /// <exception cref="ArgumentException">The memory is not one this optimizer keeps, or not of this parameter's shape.</exception>
    /// <remarks>
    /// The other half of <see cref="MemoryOf"/>: what that wrote, this reads, and a memory that is not what this optimizer
    /// writes is refused with the name of what is missing, so a run is never gone on from under another optimizer's memory.
    /// </remarks>
    protected abstract void Recall(Parameter parameter, SlotMemory memory);

    /// <summary>What the optimizer remembers of a parameter, for a checkpoint to keep.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What <see cref="MemoryOf"/> says.</returns>
    internal SlotMemory? KeptOf(Parameter parameter) => MemoryOf(parameter);

    /// <summary>Puts back what a checkpoint kept of a parameter.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <param name="memory">What was kept of it.</param>
    internal void PutBack(Parameter parameter, SlotMemory memory) => Recall(parameter, memory);

    /// <summary>A tensor of a remembered set, of the parameter's shape, or the refusal that says why not.</summary>
    /// <param name="parameter">The parameter the memory is of.</param>
    /// <param name="memory">What was remembered.</param>
    /// <param name="name">The name the tensor is kept under.</param>
    /// <returns>The tensor.</returns>
    /// <exception cref="ArgumentException">The memory holds no tensor of that name, or not of the parameter's shape.</exception>
    protected static Tensor Remembered(Parameter parameter, SlotMemory memory, string name) =>
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
    protected static Tensor Scalar(ITensorBackend backend, double value) => backend.Fill(new Shape(), (float)value);

    /// <summary>Two rates running means forget at, each a share from nothing to below one.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either is not.</exception>
    private protected static Betas RequireBetas(Betas value) =>
        value.First is >= 0 and < 1 && value.Second is >= 0 and < 1
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Each beta is a share from nothing to below one.");

    /// <summary>What is added so nothing is divided by nothing: a number above nothing.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not.</exception>
    private protected static double RequireEpsilon(double value) =>
        double.IsFinite(value) && value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Epsilon is a number above nothing.");

    /// <summary>A setting that is a number of at least nothing.</summary>
    /// <param name="value">The number.</param>
    /// <param name="what">What the setting is, as the refusal names it.</param>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    private protected static double RequireAtLeastNothing(double value, string what) =>
        double.IsFinite(value) && value >= 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, $"{what} is a number of at least nothing.");

    /// <summary>The two betas a kind of Adam's family was written with, as a list of two numbers.</summary>
    /// <exception cref="FormatException">They are missing, or not two numbers.</exception>
    private protected static Betas BetasIn(JsonElement settings, Rebuilding rebuilding)
    {
        var betas = rebuilding.Numbers(settings, "betas");

        return betas.Count == 2
            ? new Betas(betas[0], betas[1])
            : throw new FormatException("'betas' is the two rates Adam's running means forget at, as a list of two numbers.");
    }

    /// <summary>Writes the two betas of a kind of Adam's family, as a list of two numbers.</summary>
    private protected static void WriteBetas(Utf8JsonWriter writer, Betas betas)
    {
        writer.WriteStartArray("betas");
        writer.WriteNumberValue(betas.First);
        writer.WriteNumberValue(betas.Second);
        writer.WriteEndArray();
    }

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
    protected override SlotMemory? MemoryOf(Parameter parameter) =>
        _velocities.TryGetValue(parameter, out var velocity)
            ? new SlotMemory(0, new Dictionary<string, Tensor> { ["momentum_buffer"] = velocity })
            : null;

    /// <inheritdoc />
    protected override void Recall(Parameter parameter, SlotMemory memory) =>
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
    private readonly RunningMoments _moments = new();
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
        init => _betas = RequireBetas(value);
    }

    /// <summary>What is added to the root of the mean of the squares, so nothing is divided by nothing; a hundred-millionth, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number above nothing.</exception>
    public double Epsilon
    {
        get => _epsilon;
        init => _epsilon = RequireEpsilon(value);
    }

    /// <inheritdoc />
    public static string Name => "adam";

    /// <inheritdoc />
    public static Adam Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var betas = BetasIn(settings, rebuilding);

        return new(rebuilding.Number(settings, "rate")) { Betas = betas, Epsilon = rebuilding.Number(settings, "epsilon") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("rate", Rate);
        WriteBetas(writer, Betas);
        writer.WriteNumber("epsilon", Epsilon);
    }

    /// <summary>The steps a parameter has taken and its two running means.</summary>
    internal AdamMoments MomentsOf(Parameter parameter) => _moments[parameter];

    /// <inheritdoc />
    protected override SlotMemory? MemoryOf(Parameter parameter) => _moments.MemoryOf(parameter);

    /// <inheritdoc />
    protected override void Recall(Parameter parameter, SlotMemory memory) =>
        _moments.Keep(parameter, new AdamMoments(
            RunningMoments.Counted(parameter, memory, "Adam"), Remembered(parameter, memory, "exp_avg"), Remembered(parameter, memory, "exp_avg_sq")));

    /// <inheritdoc />
    protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend) =>
        _moments.After(parameter, gradient, Betas, backend).Step(parameter.Value, rate, new AdamTerms(Betas, Epsilon), backend);
}

/// <summary>
/// AdamW: Adam, with every parameter shrunk towards nothing before its step — by the rate times the weight decay, apart from
/// its gradient. PyTorch's <c>AdamW</c>, its arithmetic in PyTorch's order.
/// </summary>
/// <remarks>
/// The decay is decoupled, as Loshchilov and Hutter have it and as PyTorch's AdamW decays: the parameter is scaled by one
/// less the step's rate times the decay, and Adam's step is taken from there, so a schedule that lowers the rate lowers the
/// decay with it, and a parameter whose gradient is nothing shrinks all the same. Without a decay it steps as Adam does,
/// to the last bit. AMSGrad is not built.
/// </remarks>
public sealed class AdamW : Optimizer, ISaved<AdamW>
{
    private readonly RunningMoments _moments = new();
    private readonly Betas _betas = new(0.9, 0.999);
    private readonly double _epsilon = 1e-8;
    private readonly double _weightDecay = 0.01;

    /// <summary>AdamW at the given rate.</summary>
    /// <param name="rate">How far a step moves the parameters; a thousandth, unless said, as PyTorch leaves it.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a number above nothing.</exception>
    public AdamW(double rate = 0.001)
        : base(rate)
    {
    }

    /// <summary>How much of each running mean carries on from one step to the next; nine tenths and 0.999, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either is not a share from nothing to below one.</exception>
    public Betas Betas
    {
        get => _betas;
        init => _betas = RequireBetas(value);
    }

    /// <summary>What is added to the root of the mean of the squares, so nothing is divided by nothing; a hundred-millionth, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number above nothing.</exception>
    public double Epsilon
    {
        get => _epsilon;
        init => _epsilon = RequireEpsilon(value);
    }

    /// <summary>
    /// How far every parameter is shrunk towards nothing at each step, as a share of the step's rate; a hundredth, unless
    /// said, as PyTorch leaves it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    public double WeightDecay
    {
        get => _weightDecay;
        init => _weightDecay = RequireAtLeastNothing(value, "Weight decay");
    }

    /// <inheritdoc />
    public static string Name => "adamw";

    /// <inheritdoc />
    public static AdamW Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var betas = BetasIn(settings, rebuilding);

        return new(rebuilding.Number(settings, "rate"))
        {
            Betas = betas,
            Epsilon = rebuilding.Number(settings, "epsilon"),
            WeightDecay = rebuilding.Number(settings, "weightDecay"),
        };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("rate", Rate);
        WriteBetas(writer, Betas);
        writer.WriteNumber("epsilon", Epsilon);
        writer.WriteNumber("weightDecay", WeightDecay);
    }

    /// <inheritdoc />
    protected override SlotMemory? MemoryOf(Parameter parameter) => _moments.MemoryOf(parameter);

    /// <inheritdoc />
    protected override void Recall(Parameter parameter, SlotMemory memory) =>
        _moments.Keep(parameter, new AdamMoments(
            RunningMoments.Counted(parameter, memory, "AdamW"), Remembered(parameter, memory, "exp_avg"), Remembered(parameter, memory, "exp_avg_sq")));

    /// <inheritdoc />
    protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend)
    {
        // PyTorch's order: the parameter shrunk first — not at all without a decay, as PyTorch skips it — then Adam's step.
        var decayed = WeightDecay is 0 ? parameter.Value : backend.Scale(parameter.Value, Scalar(backend, 1 - (rate * WeightDecay)));

        return _moments.After(parameter, gradient, Betas, backend).Step(decayed, rate, new AdamTerms(Betas, Epsilon), backend);
    }
}

/// <summary>
/// RMSprop: every parameter moved by its gradient over the root of a running mean of the squares of its gradients — or,
/// with momentum, by a buffer that carries those quotients on. PyTorch's <c>RMSprop</c>, its arithmetic in PyTorch's order.
/// </summary>
/// <remarks>
/// The epsilon is added to the root, not under it, and the buffer starts at nothing, as PyTorch has them; the centred form,
/// which also keeps a running mean of the gradients, is not built.
/// </remarks>
public sealed class RmsProp : Optimizer, ISaved<RmsProp>
{
    private readonly Dictionary<Parameter, SquareMoments> _squares = [];
    private readonly double _alpha = 0.99;
    private readonly double _epsilon = 1e-8;
    private readonly double _momentum;

    /// <summary>RMSprop at the given rate.</summary>
    /// <param name="rate">How far a step moves the parameters; a hundredth, unless said, as PyTorch leaves it.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a number above nothing.</exception>
    public RmsProp(double rate = 0.01)
        : base(rate)
    {
    }

    /// <summary>How much of the running mean of the squares carries on from one step to the next; 0.99, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a share from nothing to below one.</exception>
    public double Alpha
    {
        get => _alpha;
        init => _alpha = value is >= 0 and < 1
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Alpha is a share from nothing to below one.");
    }

    /// <summary>What is added to the root of the mean of the squares, so nothing is divided by nothing; a hundred-millionth, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number above nothing.</exception>
    public double Epsilon
    {
        get => _epsilon;
        init => _epsilon = RequireEpsilon(value);
    }

    /// <summary>How much of the buffer carries on from one step to the next; none, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    public double Momentum
    {
        get => _momentum;
        init => _momentum = RequireAtLeastNothing(value, "Momentum");
    }

    /// <inheritdoc />
    public static string Name => "rmsprop";

    /// <inheritdoc />
    public static RmsProp Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "rate"))
        {
            Alpha = rebuilding.Number(settings, "alpha"),
            Epsilon = rebuilding.Number(settings, "epsilon"),
            Momentum = rebuilding.Number(settings, "momentum"),
        };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("rate", Rate);
        writer.WriteNumber("alpha", Alpha);
        writer.WriteNumber("epsilon", Epsilon);
        writer.WriteNumber("momentum", Momentum);
    }

    /// <inheritdoc />
    protected override SlotMemory? MemoryOf(Parameter parameter)
    {
        if (!_squares.TryGetValue(parameter, out var kept))
        {
            return null;
        }

        var tensors = new Dictionary<string, Tensor> { ["square_avg"] = kept.SquareAverage };

        if (kept.Buffer is { } buffer)
        {
            tensors["momentum_buffer"] = buffer;
        }

        return new SlotMemory(kept.Steps, tensors);
    }

    /// <inheritdoc />
    protected override void Recall(Parameter parameter, SlotMemory memory) =>
        _squares[parameter] = new SquareMoments(
            memory.Steps, Remembered(parameter, memory, "square_avg"), Momentum > 0 ? Remembered(parameter, memory, "momentum_buffer") : null);

    /// <inheritdoc />
    protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend)
    {
        var before = _squares.TryGetValue(parameter, out var kept) ? kept : new SquareMoments(0, backend.Fill(gradient.Shape, 0f), null);
        var squares = backend.Add(
            backend.Scale(before.SquareAverage, Scalar(backend, Alpha)),
            backend.Scale(backend.Multiply(gradient, gradient), Scalar(backend, 1 - Alpha)));
        var root = backend.Add(backend.Sqrt(squares), backend.Fill(gradient.Shape, (float)Epsilon));
        var quotient = backend.Divide(gradient, root);

        // With momentum the buffer carries the quotients on, from nothing; without, the quotient is the step.
        var direction = Momentum > 0
            ? before.Buffer is { } buffer ? backend.Add(backend.Scale(buffer, Scalar(backend, Momentum)), quotient) : quotient
            : quotient;

        _squares[parameter] = new SquareMoments(before.Steps + 1, squares, Momentum > 0 ? direction : null);

        return backend.Subtract(parameter.Value, backend.Scale(direction, Scalar(backend, rate)));
    }
}

/// <summary>
/// NAdam: Adam with Nesterov's momentum — the step taken from where the running mean is about to carry the parameter, the
/// momentum warmed up over the steps by its decay. PyTorch's <c>NAdam</c>, its arithmetic in PyTorch's order.
/// </summary>
/// <remarks>
/// Beside Adam's two running means PyTorch keeps the product of the momentums of every step so far, which is a function of
/// the steps alone: it is kept here while stepping, and worked out again from the steps a checkpoint counts, to the same
/// float, so a checkpoint holds nothing but tensors of the parameter's shape and the steps.
/// </remarks>
public sealed class Nadam : Optimizer, ISaved<Nadam>
{
    private readonly RunningMoments _moments = new();
    private readonly Dictionary<Parameter, float> _products = [];
    private readonly Betas _betas = new(0.9, 0.999);
    private readonly double _epsilon = 1e-8;
    private readonly double _momentumDecay = 0.004;

    /// <summary>NAdam at the given rate.</summary>
    /// <param name="rate">How far a step moves the parameters; two thousandths, unless said, as PyTorch leaves it.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a number above nothing.</exception>
    public Nadam(double rate = 0.002)
        : base(rate)
    {
    }

    /// <summary>How much of each running mean carries on from one step to the next; nine tenths and 0.999, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either is not a share from nothing to below one.</exception>
    public Betas Betas
    {
        get => _betas;
        init => _betas = RequireBetas(value);
    }

    /// <summary>What is added to the root of the mean of the squares, so nothing is divided by nothing; a hundred-millionth, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number above nothing.</exception>
    public double Epsilon
    {
        get => _epsilon;
        init => _epsilon = RequireEpsilon(value);
    }

    /// <summary>How fast the momentum warms up to its first beta over the steps; four thousandths, unless said, as PyTorch leaves it.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    public double MomentumDecay
    {
        get => _momentumDecay;
        init => _momentumDecay = RequireAtLeastNothing(value, "The momentum's decay");
    }

    /// <inheritdoc />
    public static string Name => "nadam";

    /// <inheritdoc />
    public static Nadam Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var betas = BetasIn(settings, rebuilding);

        return new(rebuilding.Number(settings, "rate"))
        {
            Betas = betas,
            Epsilon = rebuilding.Number(settings, "epsilon"),
            MomentumDecay = rebuilding.Number(settings, "momentumDecay"),
        };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("rate", Rate);
        WriteBetas(writer, Betas);
        writer.WriteNumber("epsilon", Epsilon);
        writer.WriteNumber("momentumDecay", MomentumDecay);
    }

    /// <summary>The product of the momentums of every step a parameter has taken, as PyTorch keeps it: a float.</summary>
    internal float MuProductOf(Parameter parameter) => _products[parameter];

    /// <inheritdoc />
    protected override SlotMemory? MemoryOf(Parameter parameter) => _moments.MemoryOf(parameter);

    /// <inheritdoc />
    protected override void Recall(Parameter parameter, SlotMemory memory)
    {
        var steps = RunningMoments.Counted(parameter, memory, "NAdam");

        _moments.Keep(parameter, new AdamMoments(steps, Remembered(parameter, memory, "exp_avg"), Remembered(parameter, memory, "exp_avg_sq")));

        // The product, worked out step by step as stepping worked it out.
        var product = 1f;

        for (var step = 1; step <= steps; step++)
        {
            product *= (float)Mu(step);
        }

        _products[parameter] = product;
    }

    /// <inheritdoc />
    protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend)
    {
        var moments = _moments.After(parameter, gradient, Betas, backend);
        var mu = Mu(moments.Steps);
        var next = Mu(moments.Steps + 1);
        var product = (_products.TryGetValue(parameter, out var kept) ? kept : 1f) * (float)mu;
        var productNext = product * next;

        _products[parameter] = product;

        // PyTorch's order: the root of the corrected mean of the squares, plus epsilon, divides the gradient and the mean in
        // two steps, each scaled by its share of the rate.
        var denominator = backend.Add(
            backend.Sqrt(backend.Divide(moments.SquaredAverage, backend.Fill(gradient.Shape, (float)(1 - Math.Pow(Betas.Second, moments.Steps))))),
            backend.Fill(gradient.Shape, (float)Epsilon));
        var first = backend.Subtract(
            parameter.Value, backend.Scale(backend.Divide(gradient, denominator), Scalar(backend, rate * (1 - mu) / (1 - product))));

        return backend.Subtract(first, backend.Scale(backend.Divide(moments.Average, denominator), Scalar(backend, rate * next / (1 - productNext))));
    }

    // The momentum of a step, warmed up towards the first beta by the momentum's decay.
    private double Mu(int step) => Betas.First * (1 - (0.5 * Math.Pow(0.96, step * MomentumDecay)));
}

/// <summary>What Adam remembers of one parameter: how many steps it took, and its running means of the gradients and of their squares.</summary>
internal readonly record struct AdamMoments(int Steps, Tensor Average, Tensor SquaredAverage)
{
    /// <summary>
    /// Where Adam's step from these means moves a parameter's tensor, in PyTorch's order: the root of the squares over the root
    /// of their correction, plus epsilon, divides the mean, and the step is the rate over the mean's correction.
    /// </summary>
    /// <param name="value">The tensor the step is taken from.</param>
    /// <param name="rate">How far the step moves it.</param>
    /// <param name="terms">The betas and the epsilon of the optimizer stepping.</param>
    /// <param name="backend">The backend the arithmetic runs on.</param>
    /// <returns>The tensor after the step.</returns>
    public Tensor Step(Tensor value, double rate, AdamTerms terms, ITensorBackend backend)
    {
        var denominator = backend.Add(
            backend.Scale(backend.Sqrt(SquaredAverage), backend.Fill(new Shape(), (float)(1 / Math.Sqrt(1 - Math.Pow(terms.Betas.Second, Steps))))),
            backend.Fill(value.Shape, (float)terms.Epsilon));
        var size = rate / (1 - Math.Pow(terms.Betas.First, Steps));

        return backend.Subtract(value, backend.Scale(backend.Divide(Average, denominator), backend.Fill(new Shape(), (float)size)));
    }
}

/// <summary>The betas and the epsilon an optimizer of Adam's family takes its step with.</summary>
/// <param name="Betas">The two rates its running means forget at.</param>
/// <param name="Epsilon">What is added to the root of the mean of the squares.</param>
internal readonly record struct AdamTerms(Betas Betas, double Epsilon);

/// <summary>
/// The running means of the gradients and of their squares an optimizer of Adam's family keeps of every parameter it moves,
/// worked out as Adam works them out, in one place for the three of them.
/// </summary>
internal sealed class RunningMoments
{
    private readonly Dictionary<Parameter, AdamMoments> _kept = [];

    /// <summary>The steps a parameter has taken and its two running means.</summary>
    /// <param name="parameter">The parameter.</param>
    public AdamMoments this[Parameter parameter] => _kept[parameter];

    /// <summary>
    /// A parameter's means after one more step with this gradient — each the last times its beta plus the gradient, or its
    /// square, times one less it, from nothing at the first step — kept, and handed back.
    /// </summary>
    /// <param name="parameter">The parameter.</param>
    /// <param name="gradient">Its gradient.</param>
    /// <param name="betas">The two rates the means forget at.</param>
    /// <param name="backend">The backend the arithmetic runs on.</param>
    /// <returns>The steps, one more, and the two means.</returns>
    public AdamMoments After(Parameter parameter, Tensor gradient, Betas betas, ITensorBackend backend)
    {
        var before = _kept.TryGetValue(parameter, out var kept)
            ? kept
            : new AdamMoments(0, backend.Fill(gradient.Shape, 0f), backend.Fill(gradient.Shape, 0f));

        var average = backend.Add(
            backend.Scale(before.Average, backend.Fill(new Shape(), (float)betas.First)), backend.Scale(gradient, backend.Fill(new Shape(), (float)(1 - betas.First))));
        var squared = backend.Add(
            backend.Scale(before.SquaredAverage, backend.Fill(new Shape(), (float)betas.Second)),
            backend.Scale(backend.Multiply(gradient, gradient), backend.Fill(new Shape(), (float)(1 - betas.Second))));

        return _kept[parameter] = new AdamMoments(before.Steps + 1, average, squared);
    }

    /// <summary>What is remembered of a parameter, under PyTorch's names; nothing before its first step.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>Its steps and its two means, or nothing.</returns>
    public SlotMemory? MemoryOf(Parameter parameter) =>
        _kept.TryGetValue(parameter, out var moments)
            ? new SlotMemory(moments.Steps, new Dictionary<string, Tensor> { ["exp_avg"] = moments.Average, ["exp_avg_sq"] = moments.SquaredAverage })
            : null;

    /// <summary>Keeps what a checkpoint says was remembered of a parameter.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <param name="moments">Its steps and its two means.</param>
    public void Keep(Parameter parameter, AdamMoments moments) => _kept[parameter] = moments;

    /// <summary>The steps a memory counts, which an optimizer of Adam's family remembers only from the first on.</summary>
    /// <param name="parameter">The parameter the memory is of.</param>
    /// <param name="memory">The memory.</param>
    /// <param name="optimizer">The optimizer's name, as the refusal says it.</param>
    /// <returns>The steps.</returns>
    /// <exception cref="ArgumentException">It counts no step.</exception>
    public static int Counted(Parameter parameter, SlotMemory memory, string optimizer) =>
        memory.Steps >= 1
            ? memory.Steps
            : throw new ArgumentException(
                $"What the checkpoint remembers of '{parameter.Name}' counts no step, and {optimizer} remembers only what stepped.", nameof(memory));
}

/// <summary>What RMSprop remembers of one parameter: how many steps it took, its running mean of the squares, and its buffer when it has momentum.</summary>
internal readonly record struct SquareMoments(int Steps, Tensor SquareAverage, Tensor? Buffer);
