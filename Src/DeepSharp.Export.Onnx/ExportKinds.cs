// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>Every kind of layer that is written, and what is said of each layer that is not.</summary>
internal static class ExportKinds
{
    // Every kind written, by the layer's class. A layer of a class not listed here is refused by name.
    private static readonly Dictionary<Type, ExportKind> Written = new ExportKind[]
    {
        new DenseKind(),
        new OperatorKind<Relu>("Relu", 6),
        new OperatorKind<Tanh>("Tanh", 6),
        new OperatorKind<Sigmoid>("Sigmoid", 6),
        new OperatorKind<Selu>("Selu", 6),
        new OperatorKind<Mish>("Mish", 18),
        new LeakyReluKind(),
        new EluKind(),
        new GeluKind(),
        new SiluKind(),
        new PReluKind(),
        new SoftmaxKind(),
        new LayerNormKind(),
        new RmsNormKind(),
        new BatchNormKind(),
        new SumKind(),
        new ConcatenationKind(),
        new ColumnsKind(),
        new StackKind(),
        new OmittedKind<Dropout>(),
        new OmittedKind<RandomFlip>(),
        new OmittedKind<RandomCrop>(),
        new OmittedKind<GaussianNoise>(),
    }.ToDictionary(kind => kind.Layer);

    /// <summary>The kind that writes a layer.</summary>
    /// <param name="layer">The layer.</param>
    /// <returns>The kind.</returns>
    /// <exception cref="NotSupportedException">No kind writes the layer, and the reason why.</exception>
    public static ExportKind For(Layer layer) =>
        Written.TryGetValue(layer.GetType(), out var kind)
            ? kind
            : throw new NotSupportedException($"A {layer.GetType().Name} is written as no ONNX node here.");
}
