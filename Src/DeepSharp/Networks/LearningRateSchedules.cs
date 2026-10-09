// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// How far each epoch's steps move the weights: the rate an optimizer starts at, and how it changes from one epoch to
/// the next.
/// </summary>
/// <remarks>
/// A schedule is asked once an epoch, and its answer is worked out from the epoch alone — a closed form, as PyTorch's own
/// schedulers can be written — so a run resumed at an epoch takes the rate it would have taken, with nothing carried from
/// the epochs before it. Every schedule here falls with the epochs, and none listens to the validation loss: a schedule
/// that lowered the rate when the validation rows stopped improving would let those rows shape the weights, where they
/// are only there to choose between them.
/// </remarks>
public abstract class LearningRateSchedule
{
    /// <summary>The rate the steps of one epoch take.</summary>
    /// <param name="epoch">The epoch, counted from nought.</param>
    /// <param name="initial">The rate the optimizer starts at.</param>
    /// <returns>The rate for that epoch.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The epoch is before the first, or the rate is not a number above nothing.</exception>
    public double RateAt(int epoch, double initial)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(epoch);

        if (!double.IsFinite(initial) || initial <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initial), initial, "A learning rate is a number above nothing.");
        }

        return Rate(epoch, initial);
    }

    /// <summary>The rate for an epoch that is one, from a rate that is one.</summary>
    /// <param name="epoch">The epoch, counted from nought.</param>
    /// <param name="initial">The rate the optimizer starts at, above nothing.</param>
    /// <returns>The rate for that epoch.</returns>
    protected abstract double Rate(int epoch, double initial);

    /// <summary>Refuses a factor a rate could not be multiplied by and stay a rate.</summary>
    /// <param name="factor">The factor.</param>
    /// <param name="parameter">The parameter it was handed in as.</param>
    /// <returns>The factor.</returns>
    private protected static double RequireFactor(double factor, string parameter) =>
        double.IsFinite(factor) && factor > 0
            ? factor
            : throw new ArgumentOutOfRangeException(parameter, factor, "A rate is multiplied by a number above nothing.");
}

/// <summary>The rate the optimizer starts at, every epoch.</summary>
public sealed class ConstantRate : LearningRateSchedule, ISaved<ConstantRate>
{
    /// <inheritdoc />
    public static string Name => "constant";

    /// <inheritdoc />
    public static ConstantRate Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override double Rate(int epoch, double initial) => initial;
}

/// <summary>The rate multiplied by a factor every so many epochs: PyTorch's <c>StepLR</c>.</summary>
public sealed class StepDecay : LearningRateSchedule, ISaved<StepDecay>
{
    /// <summary>Declares a rate that falls by a factor every so many epochs.</summary>
    /// <param name="every">How many epochs pass between two falls.</param>
    /// <param name="factor">What the rate is multiplied by at each fall; a tenth, unless said, as PyTorch leaves it.</param>
    /// <exception cref="ArgumentOutOfRangeException">It falls every fewer than one epoch, or the factor is not a number above nothing.</exception>
    public StepDecay(int every, double factor = 0.1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(every, 1);

        Every = every;
        Factor = RequireFactor(factor, nameof(factor));
    }

    /// <summary>How many epochs pass between two falls.</summary>
    public int Every { get; }

    /// <summary>What the rate is multiplied by at each fall.</summary>
    public double Factor { get; }

    /// <inheritdoc />
    public static string Name => "stepDecay";

    /// <inheritdoc />
    public static StepDecay Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "every"), rebuilding.Number(settings, "factor"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("every", Every);
        writer.WriteNumber("factor", Factor);
    }

    /// <inheritdoc />
    protected override double Rate(int epoch, double initial) => initial * Math.Pow(Factor, epoch / Every);
}

/// <summary>The rate multiplied by a factor every epoch: PyTorch's <c>ExponentialLR</c>.</summary>
public sealed class ExponentialDecay : LearningRateSchedule, ISaved<ExponentialDecay>
{
    /// <summary>Declares a rate that falls by a factor every epoch.</summary>
    /// <param name="factor">What the rate is multiplied by from one epoch to the next.</param>
    /// <exception cref="ArgumentOutOfRangeException">The factor is not a number above nothing.</exception>
    public ExponentialDecay(double factor) => Factor = RequireFactor(factor, nameof(factor));

    /// <summary>What the rate is multiplied by from one epoch to the next.</summary>
    public double Factor { get; }

    /// <inheritdoc />
    public static string Name => "exponentialDecay";

    /// <inheritdoc />
    public static ExponentialDecay Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "factor"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("factor", Factor);
    }

    /// <inheritdoc />
    protected override double Rate(int epoch, double initial) => initial * Math.Pow(Factor, epoch);
}

/// <summary>
/// The rate falling along half a cosine to its least over so many epochs, and rising again along the other half after
/// them: PyTorch's <c>CosineAnnealingLR</c>.
/// </summary>
public sealed class CosineDecay : LearningRateSchedule, ISaved<CosineDecay>
{
    /// <summary>Declares a rate that falls along a cosine.</summary>
    /// <param name="epochs">How many epochs the fall takes.</param>
    /// <param name="minimum">The least rate, reached at the last of them; nothing, unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The fall takes fewer than one epoch, or the least rate is not a number of at least nothing.</exception>
    public CosineDecay(int epochs, double minimum = 0)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(epochs, 1);

        if (!double.IsFinite(minimum) || minimum < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minimum), minimum, "The least rate is a number of at least nothing.");
        }

        Epochs = epochs;
        Minimum = minimum;
    }

    /// <summary>How many epochs the fall takes.</summary>
    public int Epochs { get; }

    /// <summary>The least rate.</summary>
    public double Minimum { get; }

    /// <inheritdoc />
    public static string Name => "cosineDecay";

    /// <inheritdoc />
    public static CosineDecay Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "epochs"), rebuilding.Number(settings, "minimum"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("epochs", Epochs);
        writer.WriteNumber("minimum", Minimum);
    }

    /// <inheritdoc />
    protected override double Rate(int epoch, double initial) =>
        Minimum + ((initial - Minimum) * (1 + Math.Cos(Math.PI * epoch / Epochs)) / 2);
}

/// <summary>
/// The rate rising in a straight line from a share of itself to the whole of it over so many epochs, and then held — or
/// handed over to the schedule that follows it: PyTorch's <c>LinearLR</c>, and its <c>SequentialLR</c> after it.
/// </summary>
/// <remarks>
/// The rate of an epoch of the warm-up is the optimizer's times the start, plus what is left to one times the share of the
/// warm-up gone by — PyTorch's closed form, its end the rate itself. The schedule it hands over to counts its epochs from the
/// end of the warm-up, as PyTorch's <c>SequentialLR</c> starts the next schedule at its milestone. A warm-up never starts at
/// nothing, as some do: a rate of nothing is refused everywhere here.
/// </remarks>
public sealed class LinearWarmup : LearningRateSchedule, ISaved<LinearWarmup>
{
    /// <summary>Declares a rate that warms up over so many epochs.</summary>
    /// <param name="epochs">How many epochs the warm-up takes.</param>
    /// <param name="start">The share of the rate the first epoch takes, above nothing and at most one; a third, unless said, as PyTorch leaves it.</param>
    /// <param name="then">The schedule the warm-up hands over to; the optimizer's own rate, every epoch, unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The warm-up takes fewer than one epoch, or its start is no share above nothing and at most one.</exception>
    public LinearWarmup(int epochs, double start = 1.0 / 3, LearningRateSchedule? then = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(epochs, 1);

        if (!double.IsFinite(start) || start <= 0 || start > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(start), start, "A warm-up starts at a share of the rate above nothing, and at most at the rate itself.");
        }

        Epochs = epochs;
        Start = start;
        Then = then;
    }

    /// <summary>How many epochs the warm-up takes.</summary>
    public int Epochs { get; }

    /// <summary>The share of the rate the first epoch takes.</summary>
    public double Start { get; }

    /// <summary>The schedule the warm-up hands over to; nothing when the optimizer's own rate follows it.</summary>
    public LearningRateSchedule? Then { get; }

    /// <inheritdoc />
    public static string Name => "linearWarmup";

    /// <inheritdoc />
    /// <remarks>The schedule it hands over to is rebuilt through the catalog where it stands, under <c>then</c>; a warm-up that names none holds the rate.</remarks>
    public static LinearWarmup Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var epochs = rebuilding.Whole(settings, "epochs");
        var start = rebuilding.Number(settings, "start");
        var then = settings.Member("then") is null ? null : rebuilding.Schedule(settings, "then");

        return new(epochs, start, then);
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">The schedule it hands over to is of no kind a file can name.</exception>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("epochs", Epochs);
        writer.WriteNumber("start", Start);

        if (Then is { } then)
        {
            writer.WritePropertyName("then");
            NetworkDocument.WriteKind(writer, then);
        }
    }

    /// <inheritdoc />
    protected override double Rate(int epoch, double initial) =>
        epoch < Epochs ? initial * (Start + ((1 - Start) * epoch / Epochs)) : Then?.RateAt(epoch - Epochs, initial) ?? initial;
}
