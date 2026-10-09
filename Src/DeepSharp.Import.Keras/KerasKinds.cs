// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

using DeepSharp.Networks;

namespace DeepSharp.Import.Keras;

/// <summary>
/// What one layer of a Keras model becomes here: the words it is written in, the activation it carries after them, the
/// slots its numbers go into, in the order Keras keeps them, and the sides of the window it walks, when it has one.
/// </summary>
/// <param name="Words">What it writes into a description in Keras's words; nothing for a layer that is its activation alone.</param>
/// <param name="Activation">The activation after it, by Keras's name for it: <c>linear</c> for none.</param>
internal readonly record struct KerasLayer(Action<Sequential>? Words, string Activation)
{
    /// <summary>The names of the slots its numbers go into, in the order Keras keeps its numbers.</summary>
    public IReadOnlyList<string> Slots { get; init; } = [];

    /// <summary>The length of each side of the window it walks, outermost first, for a convolution: none for any other layer.</summary>
    public int[] Sizes { get; init; } = [];

    /// <summary>
    /// The shape a number of this layer is handed to its slot in: a convolution's kernel, which Keras writes as one length
    /// for each side of its window, then the channels in and the channels out, as the window's places times the channels in
    /// by the channels out — the same numbers in the same order — when the lengths it begins with are the window's sides;
    /// any other number as the file writes it.
    /// </summary>
    public Shape LaidOut(ulong[] lengths)
    {
        var written = new Shape([.. lengths.Select(length => checked((int)length))]);

        return Sizes.Length > 0 && written.Rank == Sizes.Length + 2 && written.Axes[..Sizes.Length].SequenceEqual(Sizes)
            ? new Shape(Sizes.Aggregate(written[Sizes.Length], (rows, side) => checked(rows * side)), written[written.Rank - 1])
            : written;
    }
}

/// <summary>
/// A kind of Keras layer read here: what its settings are written into, in Keras's words, and what of them is refused.
/// </summary>
/// <remarks>The kinds listed here are every kind read; a layer of any other is refused, naming them.</remarks>
internal abstract class KerasKind
{
    // Every kind read, in the order a refusal names them.
    private static readonly KerasKind[] Covered =
    [
        new DenseKind(),
        new ConvolutionKind("Conv1D", 1, (filters, window) => description => description.Conv1D(filters, window.Line())),
        new ConvolutionKind("Conv2D", 2, (filters, window) => description => description.Conv2D(filters, window.Plane())),
        new ConvolutionKind("Conv3D", 3, (filters, window) => description => description.Conv3D(filters, window.Volume())),
        new PoolingKind("MaxPooling1D", 1, window => description => description.MaxPool1D(window.Line())),
        new PoolingKind("MaxPooling2D", 2, window => description => description.MaxPool2D(window.Plane())),
        new PoolingKind("MaxPooling3D", 3, window => description => description.MaxPool3D(window.Volume())),
        new PoolingKind("AveragePooling1D", 1, window => description => description.AvgPool1D(window.Line())),
        new PoolingKind("AveragePooling2D", 2, window => description => description.AvgPool2D(window.Plane())),
        new PoolingKind("AveragePooling3D", 3, window => description => description.AvgPool3D(window.Volume())),
        new GlobalPoolingKind("GlobalMaxPooling1D", 1, keepsAxes => description => description.GlobalMaxPool1D(keepsAxes)),
        new GlobalPoolingKind("GlobalMaxPooling2D", 2, keepsAxes => description => description.GlobalMaxPool2D(keepsAxes)),
        new GlobalPoolingKind("GlobalMaxPooling3D", 3, keepsAxes => description => description.GlobalMaxPool3D(keepsAxes)),
        new GlobalPoolingKind("GlobalAveragePooling1D", 1, keepsAxes => description => description.GlobalAvgPool1D(keepsAxes)),
        new GlobalPoolingKind("GlobalAveragePooling2D", 2, keepsAxes => description => description.GlobalAvgPool2D(keepsAxes)),
        new GlobalPoolingKind("GlobalAveragePooling3D", 3, keepsAxes => description => description.GlobalAvgPool3D(keepsAxes)),
        new BatchNormalizationKind(), new LayerNormalizationKind(), new DropoutKind(),
        new SpatialDropoutKind("SpatialDropout1D", 1, rate => description => description.SpatialDropout1D(rate)),
        new SpatialDropoutKind("SpatialDropout2D", 2, rate => description => description.SpatialDropout2D(rate)),
        new SpatialDropoutKind("SpatialDropout3D", 3, rate => description => description.SpatialDropout3D(rate)),
        new FlattenKind(), new ReshapeKind(), new ActivationKind(), new ReluKind(),
    ];

    /// <summary>Every kind read, by the class name Keras writes it under.</summary>
    public static IReadOnlyDictionary<string, KerasKind> ByName { get; } = Covered.ToDictionary(kind => kind.Name, StringComparer.Ordinal);

    /// <summary>The kinds read, named as a refusal names them: <c>Dense, Conv1D, … or ReLU</c>.</summary>
    public static string Listed { get; } = $"{string.Join(", ", Covered[..^1].Select(kind => kind.Name))} or {Covered[^1].Name}";

    /// <summary>The class name Keras writes the kind under.</summary>
    public abstract string Name { get; }

    /// <summary>Notes a fault when a layer works in another precision than single precision, as a network here does.</summary>
    /// <remarks>Keras writes a layer's precision as a policy's name, or as the name alone.</remarks>
    public static void RequireSinglePrecision(KerasSettings layer)
    {
        var policy = layer.Setting("dtype", default(JsonElement));
        var precision = policy.ValueKind == JsonValueKind.Object && policy.TryGetProperty("config", out var config) && config.TryGetProperty("name", out var name) ? name : policy;

        if (precision.ValueKind != JsonValueKind.Undefined && precision.ToString() != "float32")
        {
            layer.Refuse($"it works in {precision.ToString().Quoted()}, and a network here works in single precision, float32.");
        }
    }

    /// <summary>A layer of this kind, read from its settings; each fault of them noted there.</summary>
    public abstract KerasLayer Read(KerasSettings layer);

    /// <summary>The activation a layer carries, by Keras's name for it: <c>linear</c>, unless it says another.</summary>
    protected static string ActivationOf(KerasSettings layer) => layer.Setting("activation", "linear");

    /// <summary>Notes a fault when a dense layer or a convolution adds no bias, as one here always does.</summary>
    protected static void RequireBias(KerasSettings layer)
    {
        if (!layer.Setting("use_bias", true))
        {
            layer.Refuse("it adds no bias, and a dense layer or a convolution here always adds one.");
        }
    }

    /// <summary>Notes a fault when a layer lays its images out other than with their channels last, as images here are.</summary>
    protected static void RequireChannelsLast(KerasSettings layer) => RequireChannelsLast(layer, "images");

    /// <summary>Notes a fault when a layer lays what it walks out other than with its channels last, as series, images and volumes here are.</summary>
    /// <param name="layer">The layer's settings.</param>
    /// <param name="things">What the layer walks, plural, as the fault says it: <c>series</c>, <c>images</c> or <c>volumes</c>.</param>
    protected static void RequireChannelsLast(KerasSettings layer, string things)
    {
        var format = layer.Setting("data_format", "channels_last");

        if (format != "channels_last")
        {
            layer.Refuse($"it lays its {things} out '{format.Quoted()}', and {things} here are laid out with their channels last.");
        }
    }

    /// <summary>What a layer that walks so many axes walks, plural: <c>series</c>, <c>images</c> or <c>volumes</c>.</summary>
    /// <param name="axes">How many axes: one, two or three.</param>
    protected static string ThingsWalkedAlong(int axes) => axes switch { 1 => "series", 2 => "images", _ => "volumes" };

    /// <summary>Notes a fault when a normalisation works over another axis than the last, or learns no shift or no scale.</summary>
    protected static void RequireLastAxisShiftAndScale(KerasSettings layer)
    {
        if (layer.Written("axis") is { } axis and not ("-1" or "[-1]"))
        {
            layer.Refuse($"it normalises over axis {axis.Quoted()}, and a normalisation here works over the last axis.");
        }

        if (!layer.Setting("center", true) || !layer.Setting("scale", true))
        {
            layer.Refuse("it leaves out its shift or its scale, and a normalisation here learns both.");
        }
    }
}

/// <summary>Keras's <c>Dense</c>: a dense layer, its kernel inputs by outputs as the slot keeps it.</summary>
internal sealed class DenseKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Dense";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var units = layer.Setting<int>("units");
        RequireBias(layer);

        return new KerasLayer(description => description.Dense(units), ActivationOf(layer)) { Slots = ["weight", "bias"] };
    }
}

/// <summary>
/// Keras's <c>BatchNormalization</c>, in Keras's meaning: its momentum the share of the running statistics a batch leaves
/// as they were, and its epsilon.
/// </summary>
internal sealed class BatchNormalizationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "BatchNormalization";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireLastAxisShiftAndScale(layer);

        var momentum = layer.Setting<double>("momentum");
        var epsilon = layer.Setting<double>("epsilon");

        return new KerasLayer(description => description.BatchNorm(momentum, epsilon), "linear") { Slots = ["weight", "bias", "running_mean", "running_var"] };
    }
}

/// <summary>Keras's <c>LayerNormalization</c>, with its epsilon.</summary>
internal sealed class LayerNormalizationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "LayerNormalization";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireLastAxisShiftAndScale(layer);

        if (layer.Setting("rms_scaling", false))
        {
            layer.Refuse("it scales by the root mean square alone, which no normalisation here does.");
        }

        var epsilon = layer.Setting<double>("epsilon");

        return new KerasLayer(description => description.LayerNorm(epsilon), "linear") { Slots = ["weight", "bias"] };
    }
}

/// <summary>Keras's <c>Dropout</c>, leaving out each value on its own.</summary>
internal sealed class DropoutKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Dropout";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var rate = layer.Setting<double>("rate");

        if (layer.Written("noise_shape") is { } noise)
        {
            layer.Refuse($"it leaves values out by a noise shape, {noise.Quoted()}, and a dropout here leaves out each value on its own.");
        }

        return new KerasLayer(description => description.Dropout(rate), "linear");
    }
}

/// <summary>Keras's <c>Flatten</c>, of images laid out with their channels last.</summary>
internal sealed class FlattenKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Flatten";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireChannelsLast(layer);

        return new KerasLayer(description => description.Flatten(), "linear");
    }
}

/// <summary>Keras's <c>Reshape</c>, stating every length.</summary>
internal sealed class ReshapeKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Reshape";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var target = layer.Setting<int[]?>("target_shape");

        if (target is not null && target.Contains(-1))
        {
            layer.Refuse($"its target shape, {layer.Written("target_shape")!.Quoted()}, leaves a length to be worked out, and a reshape here states every length.");
        }

        return new KerasLayer(description => description.Reshape(new Shape(target!)), "linear");
    }
}

/// <summary>Keras's <c>Activation</c>: an activation alone.</summary>
internal sealed class ActivationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Activation";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer) => new(null, layer.Setting<string>("activation"));
}

/// <summary>Keras's <c>ReLU</c>, the plain one: no cap, no slope below nothing, no threshold.</summary>
internal sealed class ReluKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "ReLU";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        if (layer.Says("max_value") || layer.Setting("negative_slope", 0.0) != 0 || layer.Setting("threshold", 0.0) != 0)
        {
            layer.Refuse(
                "it caps its values, lets some below nothing through or starts above nothing, and a relu here keeps each value above nothing as it is and makes the rest nothing.");
        }

        return new KerasLayer(null, "relu");
    }
}
