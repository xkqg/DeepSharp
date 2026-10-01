// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

using DeepSharp.Networks;

namespace DeepSharp.Import.Keras;

/// <summary>
/// What one layer of a Keras model becomes here: the words it is written in, the activation it carries after them, the
/// slots its numbers go into, in the order Keras keeps them, and the window it walks, when it has one.
/// </summary>
/// <param name="Words">What it writes into a description in Keras's words; nothing for a layer that is its activation alone.</param>
/// <param name="Activation">The activation after it, by Keras's name for it: <c>linear</c> for none.</param>
internal readonly record struct KerasLayer(Action<Sequential>? Words, string Activation)
{
    /// <summary>The names of the slots its numbers go into, in the order Keras keeps its numbers.</summary>
    public IReadOnlyList<string> Slots { get; init; } = [];

    /// <summary>The window it walks, for a convolution.</summary>
    public Window? Window { get; init; }

    /// <summary>
    /// The shape a number of this layer is handed to its slot in: a convolution's kernel, which Keras writes as rows by
    /// columns by channels in by channels out, as the window's places times the channels in by the channels out — the same
    /// numbers in the same order — when its rows and columns are the window's; any other number as the file writes it.
    /// </summary>
    public Shape LaidOut(ulong[] lengths)
    {
        var written = new Shape([.. lengths.Select(length => checked((int)length))]);

        return Window is { } window && written.Rank == 4 && new Shape(written[0], written[1]) == new Shape(window.Height, window.Width)
            ? new Shape(written[0] * written[1] * written[2], written[3])
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
        new DenseKind(), new Conv2DKind(), new BatchNormalizationKind(), new LayerNormalizationKind(), new DropoutKind(),
        new FlattenKind(), new ReshapeKind(), new ActivationKind(), new ReluKind(),
    ];

    /// <summary>Every kind read, by the class name Keras writes it under.</summary>
    public static IReadOnlyDictionary<string, KerasKind> ByName { get; } = Covered.ToDictionary(kind => kind.Name, StringComparer.Ordinal);

    /// <summary>The kinds read, named as a refusal names them: <c>Dense, Conv2D, … or ReLU</c>.</summary>
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
    protected static void RequireChannelsLast(KerasSettings layer)
    {
        var format = layer.Setting("data_format", "channels_last");

        if (format != "channels_last")
        {
            layer.Refuse($"it lays its images out '{format.Quoted()}', and images here are laid out with their channels last.");
        }
    }

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

/// <summary>Keras's <c>Conv2D</c>: a convolution whose window walks one stride down and across alike, padded as 'valid' or 'same'.</summary>
internal sealed class Conv2DKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Conv2D";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var filters = layer.Setting<int>("filters");
        var size = layer.Pair("kernel_size");
        var strides = layer.Pair("strides");
        var dilation = layer.Pair("dilation_rate");
        var groups = layer.Setting("groups", 1);
        var padding = layer.Setting("padding", "valid");

        if (strides.Down != strides.Across)
        {
            layer.Refuse($"it strides {strides.Down} down and {strides.Across} across, and a window here walks one stride down and across alike.");
        }

        if (dilation != new Pair(1, 1))
        {
            layer.Refuse($"its window is dilated by {dilation.Down} down and {dilation.Across} across, and a window here covers neighbouring places.");
        }

        if (groups != 1)
        {
            layer.Refuse($"it splits its channels into {groups} groups, and a convolution here takes every channel of a place at once.");
        }

        if (padding is not ("valid" or "same"))
        {
            layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as 'valid' or 'same'.");
        }

        RequireChannelsLast(layer);
        RequireBias(layer);

        var window = new Window(size.Down, size.Across) { Stride = strides.Down, PaddingMode = padding == "same" ? PaddingMode.Same : PaddingMode.Stated };

        return new KerasLayer(description => description.Conv2D(filters, window), ActivationOf(layer)) { Slots = ["weight", "bias"], Window = window };
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
