// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>A tensor is a shape and the values that fill it, and nothing else. Everything a network does to
/// one is done by a backend, so the values can live on a CPU today and somewhere else later without this
/// type learning about either.</summary>
public class TensorTests
{
    [Fact]
    public void ATensorOfZeros_HoldsAsManyValuesAsItsShapeSaysAndEveryOneIsZero()
    {
        var tensor = Tensor.Zeros(new Shape(2, 3));

        Assert.Equal(new Shape(2, 3), tensor.Shape);
        Assert.Equal(6, tensor.Values.Length);
        Assert.All(tensor.Values.ToArray(), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void ATensor_KeepsTheValuesItWasGiven()
    {
        var tensor = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);

        Assert.Equal<float[]>([1f, 2f, 3f, 4f], tensor.Values.ToArray());
    }

    [Fact]
    public void ValuesThatDoNotFillTheShape_AreRefusedWhereTheyAreHandedOver()
    {
        var wrong = Assert.Throws<ArgumentException>(() => Tensor.From(new Shape(2, 3), [1f, 2f]));

        Assert.Contains("2x3", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("6", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATensorMadeFromAnArray_DoesNotChangeWhenThatArrayDoes()
    {
        // A caller who reuses a scratch buffer must not silently rewrite a tensor they already handed over.
        float[] scratch = [1f, 2f];
        var tensor = Tensor.From(new Shape(2), scratch);

        scratch[0] = 99f;

        Assert.Equal(1f, tensor.Values[0]);
    }

    [Fact]
    public void AScalarTensor_HoldsExactlyOneValue()
    {
        var loss = Tensor.From(new Shape(), [0.25f]);

        Assert.Equal(1, loss.Values.Length);
        Assert.Equal(0.25f, loss.Values[0]);
    }

    [Fact]
    public void ATensor_SaysWhatItIsWhenYouLookAtIt()
    {
        Assert.Equal("Tensor 2x3", Tensor.Zeros(new Shape(2, 3)).ToString());
    }
}
