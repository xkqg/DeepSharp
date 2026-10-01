// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// Rows as a pipeline hands them over — doubles, a row each — become the one tensor of floats a network takes, row after
/// row in their order, each double as the float nearest it: the one conversion the rows a network learns from, is measured
/// on and serves all go through.
/// </summary>
public class HandedRowsExtensionsTests
{
    [Fact]
    public void Rows_BecomeOneTensorOfFloats_RowAfterRow_EachTheFloatNearestItsDouble()
    {
        IReadOnlyList<double[]> rows = [[1.5, -2], [0.1, 0.25], [-0.75, 1]];

        var tensor = rows.Floats(width: 2);

        Assert.Equal(new Shape(3, 2), tensor.Shape);
        Assert.Equal([1.5f, -2f, 0.1f, 0.25f, -0.75f, 1f], tensor.Values.ToArray());
    }

    [Fact]
    public void NoRows_AreATensorOfNoRows_AsWideAsSaid()
    {
        IReadOnlyList<double[]> none = [];

        Assert.Equal(new Shape(0, 14), none.Floats(width: 14).Shape);
    }
}
