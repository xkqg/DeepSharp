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
        new RecurrentKind<Lstm>("LSTM", [0, 3, 1, 2]),
        new RecurrentKind<Gru>("GRU", [1, 0, 2]),
        new UnpaddedKind(),
        new EveryStepKind(),
        new OmittedKind<Dropout>(),
        new OmittedKind<RandomFlip>(),
        new OmittedKind<RandomCrop>(),
        new OmittedKind<GaussianNoise>(),
        new OmittedKind<SpatialDropout1D>(),
        new OmittedKind<SpatialDropout2D>(),
        new OmittedKind<SpatialDropout3D>(),
        new ConvolutionKind<Conv1D>((layer, extents) => Standing.Of(layer.Window, extents)),
        new ConvolutionKind<Conv2D>((layer, extents) => Standing.Of(layer.Window, extents)),
        new ConvolutionKind<Conv3D>((layer, extents) => Standing.Of(layer.Window, extents)),
        new TransposedConvolutionKind<Conv1DTranspose>(layer => Standing.GivenBack(layer.Window)),
        new TransposedConvolutionKind<Conv2DTranspose>(layer => Standing.GivenBack(layer.Window)),
        new TransposedConvolutionKind<Conv3DTranspose>(layer => Standing.GivenBack(layer.Window)),
        new PoolingKind<MaxPool1D>("MaxPool", (layer, extents) => Standing.Of(layer.Window, extents)),
        new PoolingKind<MaxPool2D>("MaxPool", (layer, extents) => Standing.Of(layer.Window, extents)),
        new PoolingKind<MaxPool3D>("MaxPool", (layer, extents) => Standing.Of(layer.Window, extents)),
        new PoolingKind<AvgPool1D>("AveragePool", (layer, extents) => Standing.Of(layer.Window, extents)),
        new PoolingKind<AvgPool2D>("AveragePool", (layer, extents) => Standing.Of(layer.Window, extents)),
        new PoolingKind<AvgPool3D>("AveragePool", (layer, extents) => Standing.Of(layer.Window, extents)),
        new GlobalPoolingKind<GlobalMaxPool1D>("GlobalMaxPool", "ReduceMax"),
        new GlobalPoolingKind<GlobalMaxPool2D>("GlobalMaxPool", "ReduceMax"),
        new GlobalPoolingKind<GlobalMaxPool3D>("GlobalMaxPool", "ReduceMax"),
        new GlobalPoolingKind<GlobalAvgPool1D>("GlobalAveragePool", "ReduceMean"),
        new GlobalPoolingKind<GlobalAvgPool2D>("GlobalAveragePool", "ReduceMean"),
        new GlobalPoolingKind<GlobalAvgPool3D>("GlobalAveragePool", "ReduceMean"),
        new PaddingKind<Pad1D>(layer => [layer.Border.Before, layer.Border.After]),
        new PaddingKind<Pad2D>(layer => [layer.Border.Down.Before, layer.Border.Across.Before, layer.Border.Down.After, layer.Border.Across.After]),
        new PaddingKind<Pad3D>(layer => [layer.Border.Deep.Before, layer.Border.Down.Before, layer.Border.Across.Before, layer.Border.Deep.After, layer.Border.Down.After, layer.Border.Across.After]),
        new AdaptivePoolingKind<AdaptiveMaxPool1D>("MaxPool"),
        new AdaptivePoolingKind<AdaptiveMaxPool2D>("MaxPool"),
        new AdaptivePoolingKind<AdaptiveMaxPool3D>("MaxPool"),
        new AdaptivePoolingKind<AdaptiveAvgPool1D>("AveragePool"),
        new AdaptivePoolingKind<AdaptiveAvgPool2D>("AveragePool"),
        new AdaptivePoolingKind<AdaptiveAvgPool3D>("AveragePool"),
        new EmbeddingKind(),
        new TokenEmbeddingKind(),
        new PeriodicKind(),
        new InstanceNormKind(),
        new GroupNormKind(),
        new FlattenKind(),
        new ReshapeKind(),
    }.ToDictionary(kind => kind.Layer);

    /// <summary>Whether a class of layer is written.</summary>
    /// <param name="layer">The layer's class.</param>
    public static bool Writes(Type layer) => Written.ContainsKey(layer);

    /// <summary>The kind that writes a layer.</summary>
    /// <param name="layer">The layer.</param>
    /// <returns>The kind.</returns>
    /// <exception cref="NotSupportedException">No kind writes the layer, and the reason why.</exception>
    public static ExportKind For(Layer layer) =>
        Written.TryGetValue(layer.GetType(), out var kind)
            ? kind
            : throw new NotSupportedException($"A {layer.GetType().Name} is a layer that is written as no ONNX node here.");
}
