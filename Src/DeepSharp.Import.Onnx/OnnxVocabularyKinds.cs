// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;

namespace DeepSharp.Import.Onnx;

/// <summary>ONNX's <c>Gelu</c>, from operator set 20: exact, or the hyperbolic tangent's approximation where it says <c>tanh</c>.</summary>
internal sealed class GeluKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Gelu";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var approximate = node.Text("approximate", "none");

        if (approximate is not ("none" or "tanh"))
        {
            node.Refuse($"it approximates the curve as '{approximate.Quoted()}', and a GELU here is exact, 'none', or the hyperbolic tangent's, 'tanh'.");
        }

        return new OnnxLayer(description => description.Gelu(approximate == "tanh"), reaching.Flow);
    }
}

/// <summary>ONNX's <c>LeakyRelu</c>, letting its alpha through below nothing: a hundredth, unless it says.</summary>
internal sealed class LeakyReluKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "LeakyRelu";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var slope = node.Number("alpha", 0.01);

        return new OnnxLayer(description => description.LeakyRelu(slope), reaching.Flow);
    }
}

/// <summary>ONNX's <c>Elu</c>, bending towards minus its alpha: one, unless it says.</summary>
internal sealed class EluKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Elu";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var alpha = node.Number("alpha", 1);

        return new OnnxLayer(description => description.Elu(alpha), reaching.Flow);
    }
}

/// <summary>ONNX's <c>Selu</c> at Klambauer et al.'s alpha and gamma, which it writes unless it says others.</summary>
internal sealed class SeluKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Selu";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var alpha = node.Number("alpha", 1.67326319217681884765625);
        var gamma = node.Number("gamma", 1.05070102214813232421875);

        // ONNX writes the constants in single precision: anything else is another curve.
        if (Math.Abs(alpha - 1.6732632423543772) > 1e-6 || Math.Abs(gamma - 1.0507009873554805) > 1e-6)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it bends at an alpha of {alpha} and scales by {gamma}, and a SELU here is Klambauer et al.'s, 1.6732632423543772 and 1.0507009873554805."));
        }

        return new OnnxLayer(description => description.Selu(), reaching.Flow);
    }
}

/// <summary>ONNX's <c>Mish</c>, from operator set 18.</summary>
internal sealed class MishKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Mish";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers) => new(description => description.Mish(), reaching.Flow);
}

/// <summary>
/// ONNX's <c>PRelu</c> of shares the graph holds: one for them all, or one for each channel — of a row, its values; of a series,
/// an image or a volume, the axis after the batch, which here comes last.
/// </summary>
internal sealed class PReluKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "PRelu";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var slope = numbers.Floats(node.Input(1));
        var count = slope.Lengths.Aggregate(1, (all, length) => all * length);

        // One share for each channel stands along one axis, the others of one place each, however the exporter writes them.
        if (slope.Lengths.Count(length => length > 1) > 1 || reaching.Flow is Flow.FlattenedImages or Flow.ImagesLast && count > 1)
        {
            node.Refuse($"its shares, '{node.Input(1).Quoted()}', are written as {Written(slope.Lengths)}, and a parametric rectifier here learns one share, or one for each channel.");

            return new OnnxLayer(null, reaching.Flow);
        }

        return new OnnxLayer(description => description.PRelu(eachChannel: count > 1), reaching.Flow)
        {
            Numbers = [new LayerNumber("weight", slope with { Lengths = [count] }, Laying.AsWritten)],
        };
    }
}

/// <summary>ONNX's <c>InstanceNormalization</c> of a series, an image or a volume, with its scale, its shift and its epsilon.</summary>
internal sealed class InstanceNormalizationKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "InstanceNormalization";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (!TakesImages(node, reaching))
        {
            return new OnnxLayer(null, reaching.Flow);
        }

        var epsilon = node.Number("epsilon", 1e-5);

        return new OnnxLayer(description => description.InstanceNorm(epsilon, default), reaching.Flow)
        {
            Numbers = Along(node, reaching, numbers, ["weight", "bias"]),
        };
    }
}

/// <summary>
/// ONNX's <c>GroupNormalization</c> as operator set 21 defines it: its channels in groups, a scale and a shift for each channel.
/// </summary>
internal sealed class GroupNormalizationKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "GroupNormalization";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (numbers.Opset is > 0 and < 21)
        {
            node.Refuse(string.Create(
                CultureInfo.InvariantCulture,
                $"the graph imports operator set {numbers.Opset}, whose GroupNormalization scales and shifts each group, and a group normalisation here scales and shifts each channel, as ONNX's does from operator set 21."));

            return new OnnxLayer(null, reaching.Flow);
        }

        if (reaching.Flow is Flow.FlattenedImages or Flow.ImagesLast)
        {
            node.Refuse("it normalises along the axis after the batch, and what reaches it does not hold its channels there.");

            return new OnnxLayer(null, reaching.Flow);
        }

        var groups = node.Whole("num_groups", 0);
        var epsilon = node.Number("epsilon", 1e-5);

        return new OnnxLayer(description => description.GroupNorm(groups, epsilon, default), reaching.Flow)
        {
            Numbers = Along(node, reaching, numbers, ["weight", "bias"]),
        };
    }
}

/// <summary>ONNX's <c>RMSNormalization</c>, from operator set 23, of a row over its last axis, with its scale and its epsilon.</summary>
internal sealed class RmsNormalizationKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "RMSNormalization";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var axis = node.Whole("axis", -1);

        if (reaching.Flow != Flow.Rows || axis is not (-1 or 1))
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it normalises from axis {axis} of what reaches it, and a root-mean-square normalisation here normalises the values of a row."));

            return new OnnxLayer(null, reaching.Flow);
        }

        var epsilon = node.Number("epsilon", 1e-5);

        return new OnnxLayer(description => description.RmsNorm(epsilon), reaching.Flow)
        {
            Numbers = Along(node, reaching, numbers, ["weight"]),
        };
    }
}
