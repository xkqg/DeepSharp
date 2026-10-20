// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tensors;

/// <summary>
/// One of Adam's steps for one parameter, taken number by number in a single pass: the two running means, the largest of the
/// squares under AMSGrad, the root and the quotient and the move — what the engine's twenty operations make, made in the same
/// order with the same roundings.
/// </summary>
/// <remarks>
/// Every operation of the twenty is a single operation of single precision on each number, so the pass is: a float times a float,
/// rounded to a float, and so on, with nothing fused and nothing kept in a wider width than the chain keeps it. The chain begins
/// at a mean of nothing and a mean of the squares of nothing, so a first step is worked out from those, a negative nought
/// included. A number that comes out as no number is left to the chain, because which of two not-a-numbers an operation keeps
/// is the instruction's to say and the chain is what the bits were defined by.
/// </remarks>
internal sealed class AdamPass(Tensor value, Tensor gradient, AdamMoments? before, AdamStepping stepping)
{
    /// <summary>Takes the step.</summary>
    /// <returns>The tensor after it and what is remembered now; nothing where a number came out as no number.</returns>
    public AdamStepped? Take()
    {
        var count = gradient.Shape.Count;
        var betas = stepping.Means.Betas;
        var steps = (before?.Steps ?? 0) + 1;
        var keepFirst = (float)betas.First;
        var takeFirst = (float)(1 - betas.First);
        var keepSecond = (float)betas.Second;
        var takeSecond = (float)(1 - betas.Second);
        var correction = (float)(1 / Math.Sqrt(1 - Math.Pow(betas.Second, steps)));
        var epsilon = (float)stepping.Terms.Epsilon;
        var size = (float)(stepping.Rate / (1 - Math.Pow(betas.First, steps)));
        var amsgrad = stepping.Means.AmsGrad;

        ReadOnlySpan<float> values = value.Values;
        ReadOnlySpan<float> gradients = gradient.Values;
        ReadOnlySpan<float> lastAverage = before is { } had ? had.Average.Values : default;
        ReadOnlySpan<float> lastSquared = before is { } kept ? kept.SquaredAverage.Values : default;
        ReadOnlySpan<float> lastLargest = before?.Largest is { } largest ? largest.Values : default;
        var moved = new float[count];
        var average = new float[count];
        var squared = new float[count];
        var biggest = amsgrad ? new float[count] : null;
        var noNumber = false;

        for (var at = 0; at < count; at++)
        {
            var g = gradients[at];
            var mean = (float)((lastAverage.IsEmpty ? 0f : lastAverage[at]) * keepFirst) + (float)(g * takeFirst);
            var square = (float)((lastSquared.IsEmpty ? 0f : lastSquared[at]) * keepSecond) + (float)((float)(g * g) * takeSecond);
            var rooted = square;

            if (biggest is not null)
            {
                // AMSGrad keeps the larger of the last largest and this square, chosen as the engine chooses: each multiplied by the one
                // or the nought that keeps it or leaves it out.
                var previous = lastLargest.IsEmpty ? 0f : lastLargest[at];
                var takes = (float)(square - previous) > 0f ? 1f : 0f;

                rooted = (float)(previous * (float)(1f - takes)) + (float)(square * takes);
                biggest[at] = rooted;
            }

            var denominator = (float)((float)MathF.Sqrt(rooted) * correction) + epsilon;
            var after = values[at] - (float)((float)(mean / denominator) * size);

            average[at] = mean;
            squared[at] = square;
            moved[at] = after;
            noNumber |= float.IsNaN(after) | float.IsNaN(mean) | float.IsNaN(square) | float.IsNaN(rooted);
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
}
