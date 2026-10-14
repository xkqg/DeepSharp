// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How a padding fills the places it adds from the example: what fills them, and how many it adds before and after each axis.</summary>
/// <param name="Filling">What fills the places: not one value.</param>
/// <param name="Sides">The places before and after each axis, a pair an axis in the axes' own order.</param>
internal readonly record struct FillPlan(Filling Filling, IReadOnlyList<int> Sides);

/// <summary>What fills the places a padding adds from the example, and how an engine puts them there.</summary>
internal static class FillingExtensions
{
    extension(Filling filling)
    {
        /// <summary>
        /// The matrix that picks the places of one padded axis from the axis: a row for every place of the padded axis, and a one
        /// in the column of the place it is taken from.
        /// </summary>
        /// <param name="length">How many places the axis holds.</param>
        /// <param name="before">How many places are added before it.</param>
        /// <param name="after">How many places are added after it.</param>
        /// <returns>The matrix: the padded axis' length by the axis'.</returns>
        /// <exception cref="ArgumentException">A mirror adds as many places as the axis holds, or a wrap more, or the filling is no way to fill places from the example.</exception>
        internal Tensor Picking(int length, int before, int after)
        {
            var most = filling switch
            {
                Filling.Mirror => length - 1,
                Filling.Wrap => length,
                Filling.Repeat => int.MaxValue,
                _ => throw new ArgumentOutOfRangeException(nameof(filling), filling, "Only a mirror, a repeat and a wrap fill places from the example."),
            };

            if (Math.Max(before, after) > most)
            {
                throw new ArgumentException(
                    filling == Filling.Mirror
                        ? $"A mirror padding adds at most {most} places at an end of an axis {length} places long, and this one adds {Math.Max(before, after)}: it mirrors the places beside the edge, not the edge itself."
                        : $"A wrap padding adds at most {most} places at an end of an axis {length} places long, and this one adds {Math.Max(before, after)}: it goes round the axis once.",
                    "input");
            }

            var places = before + length + after;
            var matrix = new float[places * length];

            for (var place = 0; place < places; place++)
            {
                matrix[(place * length) + filling.TakenFrom(place - before, length)] = 1f;
            }

            return Tensor.From(new Shape(places, length), matrix);
        }

        // The place of the axis a place of the padded axis is taken from; a place beyond either end is taken as the filling says.
        private int TakenFrom(int place, int length) => filling switch
        {
            Filling.Mirror => place < 0 ? -place : place >= length ? (2 * (length - 1)) - place : place,
            Filling.Repeat => Math.Clamp(place, 0, length - 1),
            _ => ((place % length) + length) % length,
        };
    }

    extension(ITensorBackend backend)
    {
        /// <summary>
        /// Every example with one axis put through a matrix of ones and noughts: the places of the axis picked, one for each row
        /// of the matrix.
        /// </summary>
        /// <param name="batch">The batch: an example by its axes by its channel.</param>
        /// <param name="axis">Which axis of the batch is picked from: after the example's, before the channel's.</param>
        /// <param name="picking">The matrix: the places wanted by the axis' places.</param>
        /// <returns>The batch, the axis as long as the matrix has rows.</returns>
        /// <remarks>
        /// The batch is seen as the matrix its axis' places stand along rows of, by two turns of the engine's matrix transpose,
        /// so the one matrix multiplication picks the places of every example, channel and other axis at once.
        /// </remarks>
        internal Tensor Picked(Tensor batch, int axis, Tensor picking)
        {
            var axes = batch.Shape.Axes.ToArray();
            var examples = axes[..axis].Aggregate(1, (all, each) => all * each);
            var length = axes[axis];
            var beyond = axes[(axis + 1)..].Aggregate(1, (all, each) => all * each);
            var places = picking.Shape[0];

            // examples by (the axis by what lies beyond it), turned to (the axis by what lies beyond it) by examples, laid out as the axis by the rest.
            var turned = backend.Reshape(backend.Transpose(backend.Reshape(batch, new Shape(examples, length * beyond))), new Shape(length, beyond * examples));
            var picked = backend.Reshape(backend.MatMul(picking, turned), new Shape(places * beyond, examples));

            return backend.Reshape(backend.Transpose(picked), new Shape([.. axes[..axis], places, .. axes[(axis + 1)..]]));
        }
    }
}
