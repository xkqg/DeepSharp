// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Import.Onnx;

/// <summary>
/// The name ONNX gives each operator a graph here is read from or written to: the one home of every such spelling, compiled into the
/// package that reads a graph and the package that writes one, so a graph one writes is a graph the other reads by the same names.
/// </summary>
internal static class OnnxOps
{
    /// <summary>ONNX's <c>Add</c> operator.</summary>
    public const string Add = "Add";

    /// <summary>ONNX's <c>Attention</c> operator.</summary>
    public const string Attention = "Attention";

    /// <summary>ONNX's <c>AveragePool</c> operator.</summary>
    public const string AveragePool = "AveragePool";

    /// <summary>ONNX's <c>BatchNormalization</c> operator.</summary>
    public const string BatchNormalization = "BatchNormalization";

    /// <summary>ONNX's <c>Cast</c> operator.</summary>
    public const string Cast = "Cast";

    /// <summary>ONNX's <c>Clip</c> operator.</summary>
    public const string Clip = "Clip";

    /// <summary>ONNX's <c>Concat</c> operator.</summary>
    public const string Concat = "Concat";

    /// <summary>ONNX's <c>Constant</c> operator.</summary>
    public const string Constant = "Constant";

    /// <summary>ONNX's <c>Conv</c> operator.</summary>
    public const string Conv = "Conv";

    /// <summary>ONNX's <c>ConvTranspose</c> operator.</summary>
    public const string ConvTranspose = "ConvTranspose";

    /// <summary>ONNX's <c>Div</c> operator.</summary>
    public const string Div = "Div";

    /// <summary>ONNX's <c>Elu</c> operator.</summary>
    public const string Elu = "Elu";

    /// <summary>ONNX's <c>Erf</c> operator.</summary>
    public const string Erf = "Erf";

    /// <summary>ONNX's <c>Expand</c> operator.</summary>
    public const string Expand = "Expand";

    /// <summary>ONNX's <c>Flatten</c> operator.</summary>
    public const string Flatten = "Flatten";

    /// <summary>ONNX's <c>Gather</c> operator.</summary>
    public const string Gather = "Gather";

    /// <summary>ONNX's <c>Gelu</c> operator.</summary>
    public const string Gelu = "Gelu";

    /// <summary>ONNX's <c>Gemm</c> operator.</summary>
    public const string Gemm = "Gemm";

    /// <summary>ONNX's <c>GlobalAveragePool</c> operator.</summary>
    public const string GlobalAveragePool = "GlobalAveragePool";

    /// <summary>ONNX's <c>GlobalMaxPool</c> operator.</summary>
    public const string GlobalMaxPool = "GlobalMaxPool";

    /// <summary>ONNX's <c>GroupNormalization</c> operator.</summary>
    public const string GroupNormalization = "GroupNormalization";

    /// <summary>ONNX's <c>GRU</c> operator.</summary>
    public const string Gru = "GRU";

    /// <summary>ONNX's <c>Identity</c> operator.</summary>
    public const string Identity = "Identity";

    /// <summary>ONNX's <c>InstanceNormalization</c> operator.</summary>
    public const string InstanceNormalization = "InstanceNormalization";

    /// <summary>ONNX's <c>LayerNormalization</c> operator.</summary>
    public const string LayerNormalization = "LayerNormalization";

    /// <summary>ONNX's <c>LeakyRelu</c> operator.</summary>
    public const string LeakyRelu = "LeakyRelu";

    /// <summary>ONNX's <c>LSTM</c> operator.</summary>
    public const string Lstm = "LSTM";

    /// <summary>ONNX's <c>MatMul</c> operator.</summary>
    public const string MatMul = "MatMul";

    /// <summary>ONNX's <c>Max</c> operator.</summary>
    public const string Max = "Max";

    /// <summary>ONNX's <c>MaxPool</c> operator.</summary>
    public const string MaxPool = "MaxPool";

    /// <summary>ONNX's <c>Mish</c> operator.</summary>
    public const string Mish = "Mish";

    /// <summary>ONNX's <c>Mul</c> operator.</summary>
    public const string Mul = "Mul";

    /// <summary>ONNX's <c>OneHot</c> operator.</summary>
    public const string OneHot = "OneHot";

    /// <summary>ONNX's <c>PRelu</c> operator.</summary>
    public const string PRelu = "PRelu";

    /// <summary>ONNX's <c>Pad</c> operator.</summary>
    public const string Pad = "Pad";

    /// <summary>ONNX's <c>Pow</c> operator.</summary>
    public const string Pow = "Pow";

    /// <summary>ONNX's <c>Reciprocal</c> operator.</summary>
    public const string Reciprocal = "Reciprocal";

    /// <summary>ONNX's <c>ReduceMax</c> operator.</summary>
    public const string ReduceMax = "ReduceMax";

    /// <summary>ONNX's <c>ReduceMean</c> operator.</summary>
    public const string ReduceMean = "ReduceMean";

    /// <summary>ONNX's <c>ReduceMin</c> operator.</summary>
    public const string ReduceMin = "ReduceMin";

    /// <summary>ONNX's <c>ReduceSum</c> operator.</summary>
    public const string ReduceSum = "ReduceSum";

    /// <summary>ONNX's <c>Relu</c> operator.</summary>
    public const string Relu = "Relu";

    /// <summary>ONNX's <c>Reshape</c> operator.</summary>
    public const string Reshape = "Reshape";

    /// <summary>ONNX's <c>RMSNormalization</c> operator.</summary>
    public const string RmsNormalization = "RMSNormalization";

    /// <summary>ONNX's <c>Selu</c> operator.</summary>
    public const string Selu = "Selu";

    /// <summary>ONNX's <c>Shape</c> operator.</summary>
    public const string Shape = "Shape";

    /// <summary>ONNX's <c>Sigmoid</c> operator.</summary>
    public const string Sigmoid = "Sigmoid";

    /// <summary>ONNX's <c>Sin</c> operator.</summary>
    public const string Sin = "Sin";

    /// <summary>ONNX's <c>Slice</c> operator.</summary>
    public const string Slice = "Slice";

    /// <summary>ONNX's <c>Softmax</c> operator.</summary>
    public const string Softmax = "Softmax";

    /// <summary>ONNX's <c>Softplus</c> operator.</summary>
    public const string Softplus = "Softplus";

    /// <summary>ONNX's <c>Split</c> operator.</summary>
    public const string Split = "Split";

    /// <summary>ONNX's <c>Sqrt</c> operator.</summary>
    public const string Sqrt = "Sqrt";

    /// <summary>ONNX's <c>Squeeze</c> operator.</summary>
    public const string Squeeze = "Squeeze";

    /// <summary>ONNX's <c>Sub</c> operator.</summary>
    public const string Sub = "Sub";

    /// <summary>ONNX's <c>Tanh</c> operator.</summary>
    public const string Tanh = "Tanh";

    /// <summary>ONNX's <c>Transpose</c> operator.</summary>
    public const string Transpose = "Transpose";

    /// <summary>ONNX's <c>Unsqueeze</c> operator.</summary>
    public const string Unsqueeze = "Unsqueeze";
}
