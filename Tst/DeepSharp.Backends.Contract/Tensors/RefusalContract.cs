// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// What the seam refuses is said once, public, and every engine says it first: an engine refuses the same operands as the
/// light engine, with the same exception, in the same words, naming the same argument — so a model cannot tell which engine
/// it runs on by what either lets through.
/// </summary>
/// <param name="engine">The engine held to the light one's refusals.</param>
public abstract class RefusalContract(ITensorBackend engine)
{
    private static readonly Window TwoByTwo = new(2, 2);

    private static readonly Dictionary<string, Refusal> Refusals = new()
    {
        ["Add of two shapes"] = new(engine => engine.Add(Zeros(2, 3), Zeros(3, 2)), "right", "Add needs two tensors of the same shape, and was given 2x3 and 3x2."),
        ["Add of a column and one value"] = new(engine => engine.Add(Zeros(32, 1), Zeros(1)), "right", "Add needs two tensors of the same shape, and was given 32x1 and 1."),
        ["Subtract of two shapes"] = new(engine => engine.Subtract(Zeros(2), Zeros(3)), "right", "Subtract needs two tensors of the same shape, and was given 2 and 3."),
        ["Multiply of two shapes"] = new(engine => engine.Multiply(Zeros(2), Zeros(3)), "right", "Multiply needs two tensors of the same shape, and was given 2 and 3."),
        ["Divide of two shapes"] = new(engine => engine.Divide(Zeros(2), Zeros(3)), "right", "Divide needs two tensors of the same shape, and was given 2 and 3."),
        ["Add of no left"] = new(engine => engine.Add(null!, Zeros(2)), "left"),
        ["Add of no right"] = new(engine => engine.Add(Zeros(2), null!), "right"),
        ["MatMul of a left row"] = new(engine => engine.MatMul(Zeros(3), Zeros(3, 1)), "left", "MatMul works on matrices, and was given a 3 tensor."),
        ["MatMul of a right row"] = new(engine => engine.MatMul(Zeros(1, 3), Zeros(3)), "right", "MatMul works on matrices, and was given a 3 tensor."),
        ["MatMul of matrices that do not meet"] = new(engine => engine.MatMul(Zeros(2, 3), Zeros(4, 5)), "right", "MatMul needs the left matrix as wide as the right one is tall, and was given 2x3 and 4x5."),
        ["MatMul of no left"] = new(engine => engine.MatMul(null!, Zeros(2, 2)), "left"),
        ["MatMul of no right"] = new(engine => engine.MatMul(Zeros(2, 2), null!), "right"),
        ["Transpose of a row"] = new(engine => engine.Transpose(Zeros(3)), "matrix", "Transpose works on matrices, and was given a 3 tensor."),
        ["Transpose of nothing"] = new(engine => engine.Transpose(null!), "matrix"),
        ["AddRow to a row"] = new(engine => engine.AddRow(Zeros(3), Zeros(3)), "matrix", "AddRow works on matrices, and was given a 3 tensor."),
        ["AddRow of a matrix"] = new(engine => engine.AddRow(Zeros(2, 3), Zeros(1, 3)), "row", "AddRow needs a row, a tensor of one axis, and was given a 1x3 one."),
        ["AddRow of a row too short"] = new(engine => engine.AddRow(Zeros(2, 3), Zeros(2)), "row", "AddRow needs a row as long as the matrix is wide, and was given 2x3 and 2."),
        ["AddRow of no row"] = new(engine => engine.AddRow(Zeros(2, 3), null!), "row"),
        ["SumRows of a row"] = new(engine => engine.SumRows(Zeros(3)), "matrix", "SumRows works on matrices, and was given a 3 tensor."),
        ["Mean of nothing held"] = new(engine => engine.Mean(Zeros(0)), "values", "A mean needs at least one value, and a 0 tensor holds none."),
        ["Mean of no tensor"] = new(engine => engine.Mean(null!), "values"),
        ["Scale by a row"] = new(engine => engine.Scale(Zeros(3), Zeros(1)), "factor", "Scale multiplies by one value, a tensor with no axes, and was given a 1 one."),
        ["Scale of no values"] = new(engine => engine.Scale(null!, Zeros()), "values"),
        ["Scale by no factor"] = new(engine => engine.Scale(Zeros(3), null!), "factor"),
        ["LogSoftmax of a row"] = new(engine => engine.LogSoftmax(Zeros(3)), "matrix", "LogSoftmax works on matrices, and was given a 3 tensor."),
        ["Reshape to another count"] = new(engine => engine.Reshape(Zeros(2, 3), new Shape(4)), "shape", "Reshape keeps every value, and a 2x3 tensor holds 6 where a 4 one holds 4."),
        ["Reshape of nothing"] = new(engine => engine.Reshape(null!, new Shape(4)), "values"),
        ["Unfold of a matrix"] = new(engine => engine.Unfold(Zeros(3, 3), TwoByTwo), "images", "Unfold works on a batch of images, image by row by column by channel, and was given a 3x3 one."),
        ["Unfold by a window of no height"] = new(engine => engine.Unfold(Zeros(1, 3, 3, 1), new Window(0, 2)), "window", "A window 0x2 (stride 1, padding 0) cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing."),
        ["Unfold by a window of no width"] = new(engine => engine.Unfold(Zeros(1, 3, 3, 1), new Window(2, 0)), "window", "A window 2x0 (stride 1, padding 0) cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing."),
        ["Unfold by a window of no stride"] = new(engine => engine.Unfold(Zeros(1, 3, 3, 1), new Window(2, 2) { Stride = 0 }), "window", "A window 2x2 (stride 0, padding 0) cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing."),
        ["Unfold by a window of a border below nothing"] = new(engine => engine.Unfold(Zeros(1, 3, 3, 1), new Window(2, 2) { Padding = -1 }), "window", "A window 2x2 (stride 1, padding -1) cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing."),
        ["Unfold by a window padded as 'same' and given a border besides"] = new(engine => engine.Unfold(Zeros(1, 3, 3, 1), new Window(2, 2) { Padding = 1, PaddingMode = PaddingMode.Same }), "window", "A window 2x2 (stride 1, padding 'same') works its border out from each image it stands on, and cannot be given one of 1 besides."),
        ["Fold by a window padded as 'same' and given a border besides"] = new(engine => engine.Fold(Zeros(9, 4), new Shape(1, 3, 3, 1), new Window(2, 2) { Padding = 2, PaddingMode = PaddingMode.Same }), "window", "A window 2x2 (stride 1, padding 'same') works its border out from each image it stands on, and cannot be given one of 2 besides."),
        ["Unfold by a window too tall"] = new(engine => engine.Unfold(Zeros(1, 3, 3, 1), new Window(4, 2)), "window", "A window 4x2 (stride 1, padding 0) is larger than a 1x3x3x1 image with its border."),
        ["Unfold by a window too wide"] = new(engine => engine.Unfold(Zeros(1, 3, 3, 1), new Window(2, 4)), "window", "A window 2x4 (stride 1, padding 0) is larger than a 1x3x3x1 image with its border."),
        ["Unfold of no images"] = new(engine => engine.Unfold(null!, TwoByTwo), "images"),
        ["Fold into images of another size"] = new(engine => engine.Fold(Zeros(4, 4), new Shape(1, 4, 4, 1), TwoByTwo), "patches", "Fold puts back the 9x4 patches a window 2x2 (stride 1, padding 0) takes of 1x4x4x1 images, and was given 4x4."),
        ["Fold into a shape that is not images"] = new(engine => engine.Fold(Zeros(4, 4), new Shape(1, 3, 3), TwoByTwo), "images", "Fold works on a batch of images, image by row by column by channel, and was given a 1x3x3 one."),
        ["Fold of patches of one axis"] = new(engine => engine.Fold(Zeros(16), new Shape(1, 3, 3, 1), TwoByTwo), "patches", "Fold puts back the 4x4 patches a window 2x2 (stride 1, padding 0) takes of 1x3x3x1 images, and was given 16."),
        ["Fold by a window that cannot stand"] = new(engine => engine.Fold(Zeros(4, 4), new Shape(1, 3, 3, 1), new Window(2, 2) { Stride = 0 }), "window", "A window 2x2 (stride 0, padding 0) cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing."),
        ["Fold by a window too large"] = new(engine => engine.Fold(Zeros(4, 4), new Shape(1, 3, 3, 1), new Window(5, 5)), "window", "A window 5x5 (stride 1, padding 0) is larger than a 1x3x3x1 image with its border."),
        ["Fold of no patches"] = new(engine => engine.Fold(null!, new Shape(1, 3, 3, 1), TwoByTwo), "patches"),
        ["Relu of nothing"] = new(engine => engine.Relu(null!), "values"),
        ["Positive of nothing"] = new(engine => engine.Positive(null!), "values"),
        ["Tanh of nothing"] = new(engine => engine.Tanh(null!), "values"),
        ["Sigmoid of nothing"] = new(engine => engine.Sigmoid(null!), "values"),
        ["Exp of nothing"] = new(engine => engine.Exp(null!), "values"),
        ["Log of nothing"] = new(engine => engine.Log(null!), "values"),
        ["Sqrt of nothing"] = new(engine => engine.Sqrt(null!), "values"),
        ["Softplus of nothing"] = new(engine => engine.Softplus(null!), "values"),
    };

    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> Refused => [.. Refusals.Keys];

    [Theory]
    [MemberData(nameof(Refused), MemberType = typeof(RefusalContract))]
    public void ThisEngine_RefusesWhatTheLightOneRefuses_InItsWords(string refused)
    {
        var refusal = Refusals[refused];

        var light = Assert.IsAssignableFrom<ArgumentException>(Record.Exception(() => refusal.Asked(new CpuBackend())));
        var own = Assert.IsAssignableFrom<ArgumentException>(Record.Exception(() => refusal.Asked(_backend)));

        Assert.Equal(refusal.Parameter, light.ParamName);
        Assert.Equal(refusal.Words is null ? typeof(ArgumentNullException) : typeof(ArgumentException), light.GetType());
        Assert.Equal(light.GetType(), own.GetType());
        Assert.Equal(light.ParamName, own.ParamName);
        Assert.Equal(light.Message, own.Message);

        if (refusal.Words is { } words)
        {
            Assert.Equal($"{words} (Parameter '{refusal.Parameter}')", light.Message);
        }
    }

    private static Tensor Zeros(params int[] axes) => Tensor.Zeros(new Shape(axes));

    // What is asked of an engine, the argument its refusal names, and its words: none for an argument not handed at all.
    private sealed record Refusal(Action<ITensorBackend> Asked, string Parameter, string? Words = null);
}
