// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>A leaky rectifier: ONNX's LeakyRelu with the share it lets through below nought.</summary>
internal sealed class LeakyReluKind : ExportKind<LeakyRelu>
{
    /// <inheritdoc />
    protected override Value Write(LeakyRelu layer, Writing at) =>
        at.Keeping(at.Graph.Node(at.Node("LeakyRelu", 6) with { Attributes = node => node.State("alpha", layer.NegativeSlope) }, [at.Input.Name], at.Path));
}

/// <summary>An exponential rectifier: ONNX's Elu with the alpha it bends towards.</summary>
internal sealed class EluKind : ExportKind<Elu>
{
    /// <inheritdoc />
    protected override Value Write(Elu layer, Writing at) =>
        at.Keeping(at.Graph.Node(at.Node("Elu", 6) with { Attributes = node => node.State("alpha", layer.Alpha) }, [at.Input.Name], at.Path));
}

/// <summary>A Gaussian error linear unit: ONNX's Gelu, which first appeared in the twentieth operator set, exact or by the tanh approximation.</summary>
internal sealed class GeluKind : ExportKind<Gelu>
{
    /// <inheritdoc />
    protected override Value Write(Gelu layer, Writing at) =>
        at.Keeping(at.Graph.Node(
            at.Node("Gelu", 20) with { Attributes = node => node.State("approximate", layer.Approximate ? "tanh" : "none") },
            [at.Input.Name],
            at.Path));
}

/// <summary>A sigmoid linear unit: the input times its own sigmoid, since ONNX has no operator of its own for it.</summary>
internal sealed class SiluKind : ExportKind<Silu>
{
    /// <inheritdoc />
    protected override Value Write(Silu layer, Writing at)
    {
        var sigmoid = at.Graph.Node(at.Node("Sigmoid", 6), [at.Input.Name], $"{at.Path}/Sigmoid");

        return at.Keeping(at.Graph.Node(at.Node("Mul", 7), [at.Input.Name, sigmoid], at.Path));
    }
}

/// <summary>A parametric rectifier: ONNX's PRelu with one share for every value, or one for each, from the ninth operator set where the shares broadcast.</summary>
internal sealed class PReluKind : ExportKind<PRelu>
{
    /// <inheritdoc />
    protected override Value Write(PRelu layer, Writing at)
    {
        // One share for each channel of a series, an image or a volume stands along the channels' axis, the others of one place each, as
        // ONNX broadcasts a share to the values it scales.
        int[] lengths = at.Input.Spatial && layer.Count > 1 ? [layer.Count, .. Enumerable.Repeat(1, at.Input.Each.Rank - 1)] : [layer.Count];
        var slopes = at.Graph.Number(at.Number(layer.Weight.Name), lengths, at.Input.Ordered(layer.Weight.Value.Values));

        return at.Keeping(at.Graph.Node(at.Node("PRelu", 9), [at.Input.Name, slopes], at.Path));
    }
}

/// <summary>Shares of one whole: ONNX's Softmax over the values of a row, or from the thirteenth operator set the channels of an image, where the axis is the one it says.</summary>
internal sealed class SoftmaxKind : ExportKind<Softmax>
{
    /// <inheritdoc />
    protected override Value Write(Softmax layer, Writing at)
    {
        // Before the thirteenth set a softmax flattens everything from its axis on into one row of shares, which is a row's values
        // and no more: over the channels of a series, an image or a volume the meaning is the thirteenth's.
        var since = at.Input.Each.Rank == 1 ? 1 : 13;

        return at.Keeping(at.Graph.Node(at.Node("Softmax", since) with { Attributes = node => node.State("axis", 1L) }, [at.Input.Name], at.Path));
    }
}
