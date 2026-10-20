// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// The mask of a row of ids: a one where an id is a word and a nought where it is padding, id nought, which a tokenizer writes after a
/// short text to make it as long as the longest.
/// </summary>
/// <remarks>
/// A recurrent layer takes it as the second step it works on, so a batch of texts of unequal lengths is read each as it would be read
/// alone, from the same ids the embedding reads: the padding is declared by the tokenizer, and nothing is learned of it. The mask holds no
/// numbers and has no gradient.
/// </remarks>
public sealed class Unpadded : Layer, ISaved<Unpadded>
{
    /// <inheritdoc />
    public static string Name => "unpadded";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static Unpadded Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of rows of ids.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 2)
        {
            throw new ArgumentException($"The mask of a batch of texts is made of rows of ids, a batch of rows, and was handed a {input.Shape} one.", nameof(input));
        }

        return Tensor.From(input.Shape, [.. input.Values.ToArray().Select(id => id == 0f ? 0f : 1f)]);
    }
}
