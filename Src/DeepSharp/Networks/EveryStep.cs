// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A layer that works on whole examples, run over every step of a series: the steps of every series are laid out as rows, the layer is
/// called once, and what it makes is put back as the series it came from — Keras's <c>TimeDistributed</c>.
/// </summary>
/// <remarks>
/// A layer that works on the last axis of what it is handed — a dense layer, a normalisation — needs no wrapper, and is run over every
/// step as it stands. This is for the layers that take a row: an embedding of numbers, a stack of layers described apart. The layer's
/// numbers are found under it, <c>layer</c>, and then under their own names.
/// </remarks>
public sealed class EveryStep : Layer, ISaved<EveryStep>
{
    private const string LayersKey = "layers";

    /// <summary>A layer run over every step of a series.</summary>
    /// <param name="layer">The layer; held by no other network.</param>
    /// <exception cref="ArgumentNullException">There is no layer.</exception>
    /// <exception cref="ArgumentException">The layer is held by another network already.</exception>
    public EveryStep(Layer layer)
    {
        ArgumentNullException.ThrowIfNull(layer);

        Layer = AddLayer("layer", layer);
    }

    /// <summary>The layer every step is run through.</summary>
    public Layer Layer { get; }

    /// <inheritdoc />
    public static string Name => "everyStep";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static EveryStep Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var layers = rebuilding.Layers(settings, LayersKey);

        return layers.Count == 1 ? new EveryStep(layers[0]) : throw rebuilding.Refuse(LayersKey, "the one layer every step is run through");
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartArray(LayersKey);
        NetworkDocument.WriteKind(writer, Layer);
        writer.WriteEndArray();
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of series of steps, each of one place at least.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank < 3 || input.Shape[1] < 1)
        {
            throw new ArgumentException($"A layer run over every step takes a batch of series, series by step by whatever a step holds, and was handed a {input.Shape} one.", nameof(input));
        }

        var backend = pass.Backend;
        var steps = input.Shape[1];
        var each = input.Shape.Axes[2..].ToArray();
        var made = Layer.Forward(backend.Reshape(input, new Shape([input.Shape[0] * steps, .. each])), pass);

        return backend.Reshape(made, new Shape([input.Shape[0], steps, .. made.Shape.Axes[1..].ToArray()]));
    }
}
