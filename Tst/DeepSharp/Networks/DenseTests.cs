// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A linear layer's doors, each refusing what cannot be a layer: no draws to start from, weights that are no matrix, and a bias
/// that is not a row as long as the weights are wide — and a bias left out, which is a layer all the same.
/// </summary>
public class DenseTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    [Fact]
    public void ALayerStartedFromNoDraws_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => new Dense(2, 3, (Draws)null!));
        Assert.Throws<ArgumentNullException>(() => new Dense(2, 3, new Beginning(null!)));
    }

    [Fact]
    public void ALayerGivenItsNumbers_NeedsWeightsThatAreAMatrix_AndABiasAsLongAsTheyAreWide()
    {
        var weights = Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]);

        Assert.Equal(
            "A linear layer's weights are a matrix and its bias a row as long as the matrix is wide, and these are 6 and no bias. (Parameter 'bias')",
            Assert.Throws<ArgumentException>(() => new Dense(Tensor.From(new Shape(6), [1f, 2f, 3f, 4f, 5f, 6f]), null)).Message);
        Assert.Contains("and 2", Assert.Throws<ArgumentException>(() => new Dense(weights, Tensor.From(new Shape(2), [1f, 2f]))).Message, StringComparison.Ordinal);

        var bare = new Dense(weights, null);
        var biased = new Dense(weights, Tensor.From(new Shape(3), [0.5f, 0.25f, 0.125f]));

        Assert.Null(bare.Bias);
        Assert.NotNull(biased.Bias);
        Assert.Equal(new Shape(3), biased.Bias.Value.Shape);
        Assert.Equal(new Shape(1, 3), bare.Forward(Tensor.From(new Shape(1, 2), [1f, 0f]), Pass.Evaluation(Engine)).Shape);
    }
}
