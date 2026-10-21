// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// The rate lowered by a factor once the training loss has stopped falling for as many epochs as the schedule waits: PyTorch's
/// <c>ReduceLROnPlateau</c> in its minimising mode, watching the loss the run trained on.
/// </summary>
/// <remarks>
/// A fall counts when the loss is below the best so far by more than the threshold, as a share of the best, which is PyTorch's
/// relative threshold. Epochs that do not count are counted, and when there have been more of them than the patience the rate is
/// multiplied by the factor, no lower than the least rate, and for as many epochs as the cooldown says the epochs that do not count
/// are not counted. A fall of less than a hundred-millionth of the rate is not made, as PyTorch leaves it.
/// <para>
/// It watches the training loss and never the validation loss: the rows a model is chosen on are not to shape the weights it is
/// chosen for. Its rate depends on the losses of the epochs before it, which a checkpoint holds, so a run that goes on from a
/// checkpoint takes the rates it would have taken — and one that goes on under another schedule is refused, since the epochs it
/// has gone through ran at rates this one would not have given. An engine that adds its totals up in another order can make a
/// loss differ in its last digit, which can move the epoch at which the rate falls: the course of a run on this schedule is the
/// engine's.
/// </para>
/// </remarks>
public sealed class TrainingPlateau : LearningRateSchedule, ISaved<TrainingPlateau>
{
    private const double SmallestFall = 1e-8;

    private double _threshold = 1e-4;
    private int _cooldown;
    private double _minimum;

    /// <summary>Declares a rate that falls once the training loss has stopped falling.</summary>
    /// <param name="patience">How many epochs without a fall are waited out before the rate is lowered; ten, unless said, as PyTorch leaves it.</param>
    /// <param name="factor">What the rate is multiplied by when it is lowered: above nothing and below one; a tenth, unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The patience is below nothing, or the factor does not lower the rate.</exception>
    public TrainingPlateau(int patience = 10, double factor = 0.1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(patience);

        if (!(factor > 0 && factor < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(factor), factor, "The rate is lowered by a factor above nothing and below one.");
        }

        Patience = patience;
        Factor = factor;
    }

    /// <summary>How many epochs without a fall are waited out before the rate is lowered.</summary>
    public int Patience { get; }

    /// <summary>What the rate is multiplied by when it is lowered.</summary>
    public double Factor { get; }

    /// <summary>How far below the best a loss has to be, as a share of the best, to count as a fall; a ten-thousandth, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number from nothing to below one.</exception>
    public double Threshold
    {
        get => _threshold;
        init
        {
            if (!(value >= 0 && value < 1))
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "A fall counts when the loss is below the best by a share of it: from nothing to below one.");
            }

            _threshold = value;
        }
    }

    /// <summary>How many epochs after a lowering the epochs without a fall are not counted; none, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing.</exception>
    public int Cooldown
    {
        get => _cooldown;
        init
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);

            _cooldown = value;
        }
    }

    /// <summary>The least the rate is lowered to; nothing, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a number.</exception>
    public double Minimum
    {
        get => _minimum;
        init
        {
            if (double.IsNaN(value) || value < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "The least rate is nothing or more.");
            }

            _minimum = value;
        }
    }

    /// <inheritdoc />
    public static string Name => "trainingPlateau";

    /// <inheritdoc />
    /// <remarks>The eighth: the first an older library refuses by its number, since a rate that follows the losses is a thing it cannot give.</remarks>
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static TrainingPlateau Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "patience"), rebuilding.Number(settings, "factor"))
        {
            Threshold = rebuilding.Number(settings, "threshold"),
            Cooldown = rebuilding.Whole(settings, "cooldown"),
            Minimum = rebuilding.Number(settings, "minimum"),
        };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("patience", Patience);
        writer.WriteNumber("factor", Factor);
        writer.WriteNumber("threshold", Threshold);
        writer.WriteNumber("cooldown", Cooldown);
        writer.WriteNumber("minimum", Minimum);
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">Always: the rate of an epoch is worked out from the losses before it, and the epoch alone says none.</exception>
    protected override double Rate(int epoch, double initial) =>
        throw new InvalidOperationException("A rate that follows the training loss is worked out from the losses of the epochs before it: ask RateAfter with them.");

    /// <inheritdoc />
    protected override double Rate(IReadOnlyList<double> trainingLosses, double initial)
    {
        var rate = initial;
        var best = double.PositiveInfinity;
        var waited = 0;
        var cooling = 0;

        foreach (var loss in trainingLosses)
        {
            if (loss < best * (1 - Threshold))
            {
                best = loss;
                waited = 0;
            }
            else
            {
                waited++;
            }

            if (cooling > 0)
            {
                cooling--;
                waited = 0;
            }

            if (waited > Patience)
            {
                var lowered = Math.Max(rate * Factor, Minimum);

                if (rate - lowered > SmallestFall)
                {
                    rate = lowered;
                }

                cooling = Cooldown;
                waited = 0;
            }
        }

        return rate;
    }
}
