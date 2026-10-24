// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Backends.TorchSharp;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.TorchGpu;
using TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// An engine that keeps its numbers in half the width: what it makes of a product is kept in that width and read back as the floats
/// the width can hold, what it does with the numbers beside the products is decided by one table, and what a tensor of another width
/// meets when it reaches it. The proof of a width is in the bits: a value kept in bfloat16 has its lower sixteen bits nought, one kept
/// in half its lower thirteen, whatever it was worked out as on the way.
/// </summary>
public class ReducedPrecisionTests
{
    private static readonly Tensor Left = Tensor.From(new Shape(3, 4), [0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f, -0.5f, 0.25f, 0.125f, 3.3f]);

    private static readonly Tensor Right = Tensor.From(new Shape(4, 2), [0.7f, 0.9f, 1.3f, -0.2f, 0.45f, 0.33f, 2.1f, 1.7f]);

    [Fact]
    public void AProductOfAnEngineKeepingBFloat16_IsKeptInBFloat16_AndReadsOutAsTheFloatsItCanHold()
    {
        var reduced = TorchBackend.OnCpu(Precision.BFloat16).MatMul(Left, Right);
        var full = TorchBackend.OnCpu().MatMul(Left, Right);

        Assert.All(reduced.Values.ToArray(), value => Assert.Equal(0, BitConverter.SingleToInt32Bits(value) & 0xFFFF));
        Assert.True(reduced.Values.ToArray().Zip(full.Values.ToArray()).All(pair => Math.Abs(pair.First - pair.Second) <= 0.01 * (Math.Abs(pair.Second) + 1)));
        Assert.Contains(reduced.Values.ToArray().Zip(full.Values.ToArray()), pair => BitConverter.SingleToInt32Bits(pair.First) != BitConverter.SingleToInt32Bits(pair.Second));
    }

    [Fact]
    public void ABiasAddedToAReducedProduct_FollowsTheProduct_SoWhatTheLayerMakesStaysReduced()
    {
        var engine = TorchBackend.OnCpu(Precision.BFloat16);

        var made = engine.AddRow(engine.MatMul(Left, Right), Tensor.From(new Shape(2), [0.1f, -0.3f]));

        Assert.All(made.Values.ToArray(), value => Assert.Equal(0, BitConverter.SingleToInt32Bits(value) & 0xFFFF));
    }

    [Fact]
    public void AnOpOfTwoTensorsOfFullWidth_StaysFullWidth_ToTheBit()
    {
        var reduced = TorchBackend.OnCpu(Precision.BFloat16).Add(Left, Left);
        var full = TorchBackend.OnCpu().Add(Left, Left);

        Assert.Equal(full.Values.ToArray(), reduced.Values.ToArray());
        Assert.Contains(reduced.Values.ToArray(), value => (BitConverter.SingleToInt32Bits(value) & 0xFFFF) != 0);
    }

    [Fact]
    public void AReducedAndAFullTensorMeetingInASum_ComeToFullWidth_WithTheReducedOneTakenInExactly()
    {
        var engine = TorchBackend.OnCpu(Precision.BFloat16);
        var product = engine.MatMul(Left, Right);
        var third = Tensor.From(product.Shape, [.. Enumerable.Repeat(1f / 3f, product.Shape.Count)]);

        var sum = engine.Add(product, third).Values.ToArray();

        for (var at = 0; at < sum.Length; at++)
        {
            Assert.Equal(product.Values[at] + (1f / 3f), sum[at]);
        }
    }

    [Fact]
    public void AnOpTheTableCallsWide_WorksInFullWidth_OnWhatAReducedProductMade()
    {
        var engine = TorchBackend.OnCpu(Precision.BFloat16);
        var product = engine.MatMul(Left, Right);

        var exponent = engine.Exp(product).Values.ToArray();
        var expected = TorchBackend.OnCpu().Exp(product).Values.ToArray();

        Assert.Equal(expected, exponent);
        Assert.Contains(exponent, value => (BitConverter.SingleToInt32Bits(value) & 0xFFFF) != 0);
    }

    [Fact]
    public void ATensorOfAReducedEngine_HandedToAFullEngine_IsTakenInAsWhatItHolds_AndNothingIsRoundedTwice()
    {
        var product = TorchBackend.OnCpu(Precision.BFloat16).MatMul(Left, Right);

        var doubled = TorchBackend.OnCpu().Add(product, product).Values.ToArray();

        for (var at = 0; at < doubled.Length; at++)
        {
            Assert.Equal(product.Values[at] * 2, doubled[at]);
        }
    }

    [Fact]
    public void AFullEngine_IsTheEngineItWas_ToTheBit()
    {
        var named = TorchBackend.OnCpu(Precision.Full).MatMul(Left, Right).Values.ToArray();
        var before = TorchBackend.OnCpu().MatMul(Left, Right).Values.ToArray();

        Assert.Equal(before.Select(BitConverter.SingleToInt32Bits), named.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void AReducedStorage_CountsItsOwnBytes_TwoToAValue()
    {
        var product = TorchBackend.OnCpu(Precision.BFloat16).MatMul(Left, Right);
        var full = TorchBackend.OnCpu().MatMul(Left, Right);

        Assert.Equal(product.Shape.Count * 2L, ((TorchStorage)product.Storage).Bytes);
        Assert.Equal(full.Shape.Count * 4L, ((TorchStorage)full.Storage).Bytes);
    }

    [Fact]
    public void OwnershipIsTheDeviceAndTheFormat_NotTheTextTheEngineNamesItselfBy()
    {
        var product = TorchBackend.OnCpu(Precision.BFloat16).MatMul(Left, Right);
        var storage = (TorchStorage)product.Storage;

        Assert.True(storage.IsHeldFor("cpu", torch.ScalarType.BFloat16));
        Assert.False(storage.IsHeldFor("cpu", torch.ScalarType.Float32));
        Assert.False(storage.IsHeldFor("cuda:0", torch.ScalarType.BFloat16));
    }

    [Fact]
    public void HalfOnTheProcessor_IsRefused_NamingTheInstructionsItNeeds_AndWhatToUseInstead()
    {
        var refused = Assert.Throws<ArgumentException>(() => TorchBackend.OnCpu(Precision.Half));

        Assert.Contains("AVX512-FP16", refused.Message, StringComparison.Ordinal);
        Assert.Contains("AVX-NE-CONVERT", refused.Message, StringComparison.Ordinal);
        Assert.Contains(nameof(Precision.BFloat16), refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheSharesOfANumberReadTwice_ReachItAtFullWidth_ThoughEachWasMadeNarrow()
    {
        var engine = TorchBackend.OnCpu(Precision.BFloat16);
        var weights = Tensor.From(new Shape(4, 2), [0.7f, 0.9f, 1.3f, -0.2f, 0.45f, 0.33f, 2.1f, 1.7f]);
        var other = Tensor.From(new Shape(3, 4), [0.3f, -0.7f, 0.15f, 0.9f, 0.45f, 0.21f, -1.1f, 0.6f, 0.77f, 0.05f, 0.31f, -0.4f]);
        var pass = new RecordingBackend(engine);

        var loss = pass.Mean(pass.Add(pass.MatMul(Left, weights), pass.MatMul(other, weights)));
        var gradient = pass.GradientsOf(loss, [weights])[weights].Values.ToArray();

        // Two narrow shares added up in the narrow width would come to a value of the narrow width; added up wide they do not.
        Assert.Contains(gradient, value => (BitConverter.SingleToInt32Bits(value) & 0xFFFF) != 0);
    }

    [Fact]
    public void ARunOnAReducedEngine_KeepsTheNumbersItLearnsAtFullWidth_AfterEveryStep()
    {
        var rows = Tensor.From(new Shape(8, 3), [.. Enumerable.Range(0, 24).Select(at => (float)Math.Sin(at * 0.7))]);
        var answers = Tensor.From(new Shape(8, 1), [.. Enumerable.Range(0, 8).Select(at => (float)Math.Cos(at * 0.3))]);
        var compiled = new Sequential().Dense(4).Relu().Dense(1).Compile(new Adam(0.01), new MeanSquaredError());

        _ = compiled.Fit(new TrainingData(rows, answers), null, new FitOptions(seed: 3) { Epochs = 3, BatchSize = 4, Backend = TorchBackend.OnCpu(Precision.BFloat16) });

        var learned = compiled.Network.Parameters().SelectMany(parameter => parameter.Value.Values.ToArray()).ToArray();

        Assert.True(learned.All(float.IsFinite));
        Assert.Contains(learned, value => (BitConverter.SingleToInt32Bits(value) & 0xFFFF) != 0);
    }

    [Fact]
    public void TheSumOfTheRowsOfAMatrixInSixteenBits_WithMostOfTheCardTaken_IsMadeWithoutACopyOfItAtFullWidth()
    {
        Assert.SkipUnless(Card.Present, Card.Absent);

        var engine = TorchBackend.OnGpu(0, Precision.BFloat16);

        // Two matrices of singles of 4 GB each are held, and a product of a column of ones and a row of ones makes a matrix in sixteen bits
        // of 2,000,000 by 1024 values, 3.8 GiB. A copy of it at full width, which a sum that read its values as singles first would make beside it,
        // is 7.6 GiB more than a card of 16 GiB has left.
        var held = new[] { engine.Fill(new Shape(1_000_000, 1024), 1f), engine.Fill(new Shape(1_000_000, 1024), 1f) };
        var matrix = engine.MatMul(engine.Fill(new Shape(2_000_000, 1), 1f), engine.Fill(new Shape(1, 1024), 1f));
        var sums = engine.SumRows(matrix);

        Assert.Equal(new Shape(1024), sums.Shape);
        Assert.Equal(2_000_000f, sums.Values[0]);
        GC.KeepAlive(held);
    }

    [Fact]
    public void AProductOfAnEngineKeepingHalf_OnTheCard_IsKeptInHalf_AndTheEngineNeverSetsTf32()
    {
        Assert.SkipUnless(Card.Present, Card.Absent);

        var before = torch.backends.cuda.matmul.allow_tf32;
        var engine = TorchBackend.OnGpu(0, Precision.Half);

        var made = engine.MatMul(Left, Right).Values.ToArray();

        Assert.All(made, value => Assert.Equal(0, BitConverter.SingleToInt32Bits(value) & 0x1FFF));
        Assert.Equal(before, torch.backends.cuda.matmul.allow_tf32);
    }
}
