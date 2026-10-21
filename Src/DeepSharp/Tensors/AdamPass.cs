// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Numerics;
using DeepSharp.Networks;

namespace DeepSharp.Tensors;

/// <summary>
/// One of Adam's steps for one parameter, taken number by number in a single pass: the two running means, the largest of the
/// squares under AMSGrad, the root and the quotient and the move — what the engine's twenty operations make, made in the same
/// order with the same roundings.
/// </summary>
/// <remarks>
/// Every operation of the twenty is a single operation of single precision on each number, so the pass is: a float times a float,
/// rounded to a float, and so on, with nothing fused and nothing kept in a wider width than the chain keeps it — a vector's width of
/// numbers at a time, and the last few one at a time, which are the same operations. The chain begins at a mean of nothing and a mean
/// of the squares of nothing, so a first step is worked out from those, a negative nought included. A number that comes out as no
/// number is left to the chain, because which of two not-a-numbers an operation keeps is the instruction's to say and the chain is
/// what the bits were defined by.
/// </remarks>
internal sealed class AdamPass(Tensor gradient, AdamMoments? before, AdamStepping stepping)
{
    /// <summary>Takes the step.</summary>
    /// <returns>The tensor after it and what is remembered now; nothing where a number came out as no number.</returns>
    public AdamStepped? Take()
    {
        var count = gradient.Shape.Count;
        var betas = stepping.Means.Betas;
        var steps = (before?.Steps ?? 0) + 1;
        var terms = new Terms(
            (float)betas.First,
            (float)(1 - betas.First),
            (float)betas.Second,
            (float)(1 - betas.Second),
            (float)(1 / Math.Sqrt(1 - Math.Pow(betas.Second, steps))),
            (float)stepping.Terms.Epsilon,
            (float)(stepping.Rate / (1 - Math.Pow(betas.First, steps))));

        ReadOnlySpan<float> values = stepping.From.Values;
        ReadOnlySpan<float> gradients = gradient.Values;
        ReadOnlySpan<float> lastAverage = before is { } had ? had.Average.Values : default;
        ReadOnlySpan<float> lastSquared = before is { } kept ? kept.SquaredAverage.Values : default;
        ReadOnlySpan<float> lastLargest = before?.Largest is { } largest ? largest.Values : default;
        var moved = new float[count];
        var average = new float[count];
        var squared = new float[count];
        var biggest = stepping.Means.AmsGrad ? new float[count] : null;
        var at = 0;
        var noNumber = false;

        if (Vector.IsHardwareAccelerated)
        {
            var width = Vector<float>.Count;
            var keepFirst = new Vector<float>(terms.KeepFirst);
            var takeFirst = new Vector<float>(terms.TakeFirst);
            var keepSecond = new Vector<float>(terms.KeepSecond);
            var takeSecond = new Vector<float>(terms.TakeSecond);
            var correction = new Vector<float>(terms.Correction);
            var epsilon = new Vector<float>(terms.Epsilon);
            var size = new Vector<float>(terms.Size);

            for (; at + width <= count; at += width)
            {
                var g = new Vector<float>(gradients.Slice(at, width));
                var mean = ((lastAverage.IsEmpty ? Vector<float>.Zero : new Vector<float>(lastAverage.Slice(at, width))) * keepFirst) + (g * takeFirst);
                var square = ((lastSquared.IsEmpty ? Vector<float>.Zero : new Vector<float>(lastSquared.Slice(at, width))) * keepSecond) + ((g * g) * takeSecond);
                var rooted = square;

                if (biggest is not null)
                {
                    // AMSGrad keeps the larger of the last largest and this square, chosen as the engine chooses: each multiplied by the one
                    // or the nought that keeps it or leaves it out.
                    var previous = lastLargest.IsEmpty ? Vector<float>.Zero : new Vector<float>(lastLargest.Slice(at, width));
                    var takes = Vector.ConditionalSelect(Vector.GreaterThan(square - previous, Vector<float>.Zero), Vector<float>.One, Vector<float>.Zero);

                    rooted = (previous * (Vector<float>.One - takes)) + (square * takes);
                    rooted.CopyTo(biggest, at);
                }

                var after = new Vector<float>(values.Slice(at, width)) - ((mean / ((Vector.SquareRoot(rooted) * correction) + epsilon)) * size);

                mean.CopyTo(average, at);
                square.CopyTo(squared, at);
                after.CopyTo(moved, at);
                noNumber = noNumber || !Vector.EqualsAll(after, after) || !Vector.EqualsAll(mean, mean) || !Vector.EqualsAll(square, square) || !Vector.EqualsAll(rooted, rooted);
            }
        }

        for (; at < count; at++)
        {
            // Every product and every sum is a single of its own, rounded where it stands, as the vector path rounds each of its operations.
            var g = gradients[at];
            float keptMean = (lastAverage.IsEmpty ? 0f : lastAverage[at]) * terms.KeepFirst;
            float takenMean = g * terms.TakeFirst;
            float mean = keptMean + takenMean;
            float keptSquare = (lastSquared.IsEmpty ? 0f : lastSquared[at]) * terms.KeepSecond;
            float gradientSquare = g * g;
            float takenSquare = gradientSquare * terms.TakeSecond;
            float square = keptSquare + takenSquare;
            var rooted = square;

            if (biggest is not null)
            {
                var previous = lastLargest.IsEmpty ? 0f : lastLargest[at];
                float difference = square - previous;
                var takes = difference > 0f ? 1f : 0f;
                float left = 1f - takes;
                float keptLargest = previous * left;
                float takenLargest = square * takes;

                rooted = keptLargest + takenLargest;
                biggest[at] = rooted;
            }

            float root = MathF.Sqrt(rooted);
            float scaled = root * terms.Correction;
            float denominator = scaled + terms.Epsilon;
            float quotient = mean / denominator;
            float step = quotient * terms.Size;
            var after = values[at] - step;

            average[at] = mean;
            squared[at] = square;
            moved[at] = after;
            noNumber = noNumber || float.IsNaN(after) || float.IsNaN(mean) || float.IsNaN(square) || float.IsNaN(rooted);
        }

        if (noNumber)
        {
            return null;
        }

        var shape = gradient.Shape;

        return new AdamStepped(
            Tensor.Wrap(shape, moved),
            new AdamMoments(steps, Tensor.Wrap(shape, average), Tensor.Wrap(shape, squared), biggest is null ? null : Tensor.Wrap(shape, biggest)));
    }

    // The single-precision numbers a step is worked out with, each as the chain rounds it to a float.
    private readonly record struct Terms(float KeepFirst, float TakeFirst, float KeepSecond, float TakeSecond, float Correction, float Epsilon, float Size);
}
