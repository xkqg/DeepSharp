// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// The seam an engine implements grows by what the layers need and nothing else, and every operation it gains is one more an
/// engine written outside the library has to write. So the operations it holds are named here, and one gained or lost fails
/// this until the release that changes them says so in its upgrading notes.
/// </summary>
public class SeamTests
{
    [Fact]
    public void TheSeam_HoldsExactlyTheseOperations()
    {
        string[] operations =
        [
            "Add", "Subtract", "Multiply", "MatMul", "Transpose", "AddRow", "SumRows", "Mean", "Scale", "Fill", "Relu", "Positive",
            "FirstLargest", "Tanh", "Sigmoid", "Exp", "Log", "Sqrt", "Softplus", "Sin", "Erf", "Divide", "LogSoftmax", "Reshape",
            "Unfold", "Fold",
        ];

        Assert.Equal(
            operations.Order(StringComparer.Ordinal),
            typeof(ITensorBackend).GetMethods().Where(method => !method.IsSpecialName).Select(method => method.Name).Order(StringComparer.Ordinal));
    }
}
