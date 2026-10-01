// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using static DeepSharp.Tests.Backends.Contract.LayerContract;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A slot is what changes when learning puts a new tensor into it — by an optimizer, a restore and the reader alone, the
/// door the library keeps to itself. What a layer does on any engine runs from the contract; these reach that door.
/// </summary>
public class LayerTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void AParameter_RefusesATensorOfAnotherShape()
    {
        var parameter = new Weighted(2f, 3).Weight;

        var wrong = Assert.Throws<ArgumentException>(() => parameter.Replace(Tensor.Zeros(new Shape(4))));

        Assert.Contains("weight", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("3", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("4", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASlot_KeepsWhoItIs_WhenItsTensorIsReplaced()
    {
        var stack = new LayerStack(new Weighted(2f, 3));
        var parameter = stack.Parameters().Single();
        var replacement = Tensor.From(new Shape(3), [5f, 6f, 7f]);

        parameter.Replace(replacement);

        Assert.Same(parameter, stack.Parameters().Single());
        Assert.Same(replacement, parameter.Value);
    }

    [Fact]
    public void ASnapshot_BringsBackEverySlot_TheParametersAndTheRunningStatisticsAlike()
    {
        var network = new TwoLayers();
        var snapshot = network.Snapshot();
        var before = network.Slots().Select(slot => slot.Slot.Value).ToArray();

        network.Parameters().First().Replace(Tensor.From(new Shape(3), [9f, 9f, 9f]));
        network.Forward(Tensor.Zeros(new Shape(1, 3)), Pass.Training(_backend, new RandomStream(7), epoch: 0, step: 0));
        network.Restore(snapshot);

        Assert.Equal(before, network.Slots().Select(slot => slot.Slot.Value));
    }
}
