// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// An engine that keeps its work in a narrow width has the shares of a gradient that reach a number a network learns taken to full
/// width before they are added, so the sum of what several batches or several readings sent back is kept wider than each of them
/// was; no other tensor is asked to be widened, and an engine that cannot widen is handed nothing to widen.
/// </summary>
public class WideningTests
{
    private static readonly Tensor Days = Tensor.From(new Shape(2, 2), [0.7993437f, -0.5730045f, 0.7866853f, -0.6739606f]);

    [Fact]
    public void EachShareOfANumberReadTwice_IsWidened_BeforeTheyAreAdded_AndNothingElseIs()
    {
        var engine = new WideningBackend(new CpuBackend());
        var weights = Tensor.From(new Shape(2, 1), [0.1f, -0.2f]);
        var pass = new RecordingBackend(engine);

        var first = pass.MatMul(Days, weights);
        var second = pass.MatMul(Days, weights);
        var loss = pass.Mean(pass.Add(first, second));

        var gradients = pass.GradientsOf(loss, [weights]);

        Assert.Equal([new Shape(2, 1), new Shape(2, 1)], engine.Widened);
        Assert.Equal(2 * 0.5f * (Days.Values[0] + Days.Values[2]), gradients[weights].Values[0], 5);
    }

    [Fact]
    public void ANumberTheLossNeverReached_IsGivenNothingToWiden()
    {
        var engine = new WideningBackend(new CpuBackend());
        var weights = Tensor.From(new Shape(2, 1), [0.1f, -0.2f]);
        var aside = Tensor.From(new Shape(2, 1), [0.3f, 0.4f]);
        var pass = new RecordingBackend(engine);

        var loss = pass.Mean(pass.MatMul(Days, weights));

        _ = pass.MatMul(Days, aside);

        var gradients = pass.GradientsOf(loss, [weights]);

        Assert.Single(engine.Widened);
        Assert.Equal(new Shape(2, 1), gradients[weights].Shape);
    }

    [Fact]
    public void AnEngineThatCannotWiden_IsHandedNothingToWiden_AndWorksAsBefore()
    {
        var weights = Tensor.From(new Shape(2, 1), [0.1f, -0.2f]);
        var pass = new RecordingBackend(new CpuBackend());

        var gradients = pass.GradientsOf(pass.Mean(pass.MatMul(Days, weights)), [weights]);

        Assert.Equal(0.5f * (Days.Values[0] + Days.Values[2]), gradients[weights].Values[0], 5);
    }
}
