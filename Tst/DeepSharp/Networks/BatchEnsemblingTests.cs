// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A BatchEnsemble adapter multiplies every row of a batch by a different one of several learned rows, laid out variant
/// by variant within the place of each row that made them; variant heads close the variants again, each with its own,
/// un-shared weights.
/// </summary>
public class BatchEnsemblingTests
{
    private static Draws DrawsFor(string purpose) => new RandomStream(42).Draw(purpose, 0, 0);

    [Fact]
    public void ABatchEnsembleAdapter_MultipliesEveryRowByADifferentVariantsRow_LaidOutVariantByVariantWithinEachRowsPlace()
    {
        var layer = new BatchEnsemble(variants: 2, inputs: 2, DrawsFor("0.adapter"));

        layer.Adapter.Replace(Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]));

        var input = Tensor.From(new Shape(2, 2), [5f, 6f, 7f, 8f]);
        var output = layer.Forward(input, Pass.Evaluation(new CpuBackend()));

        Assert.Equal(new Shape(2, 2, 2), output.Shape);
        Assert.Equal([5f, 12f, 15f, 24f, 7f, 16f, 21f, 32f], output.Values.ToArray());
    }

    [Fact]
    public void ABatchEnsembleAdapter_NeedsAtLeastOneVariantAndOneInput()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchEnsemble(0, 2, DrawsFor("x")));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchEnsemble(2, 0, DrawsFor("x")));
    }

    [Fact]
    public void ABatchEnsembleAdapter_RefusesABatchOfAnotherWidth_OrOfOneRowAlone()
    {
        var layer = new BatchEnsemble(2, 2, DrawsFor("0.adapter"));

        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]), Pass.Evaluation(new CpuBackend())));
        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.From(new Shape(2), [1f, 2f]), Pass.Evaluation(new CpuBackend())));
    }

    [Fact]
    public void VariantHeads_AnswersEachRowByItsOwnVariantsWeightsAndBias_NeverAnothers()
    {
        var heads = new VariantHeads(variants: 2, inputs: 2, outputs: 1, DrawsFor("0.heads"));

        heads.Weight.Replace(Tensor.From(new Shape(4, 1), [1f, 1f, 2f, 2f]));
        heads.Bias.Replace(Tensor.From(new Shape(2, 1), [0f, 10f]));

        // Two rows of the batch, two variants each: (b=0,v=0), (b=0,v=1), (b=1,v=0), (b=1,v=1).
        var input = Tensor.From(new Shape(2, 2, 2), [1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f]);
        var output = heads.Forward(input, Pass.Evaluation(new CpuBackend()));

        Assert.Equal(new Shape(2, 2, 1), output.Shape);
        Assert.Equal([2f, 18f, 6f, 26f], output.Values.ToArray());
    }

    [Fact]
    public void VariantHeads_NeedsAtLeastOneVariantOneInputAndOneOutput()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new VariantHeads(0, 2, 2, DrawsFor("x")));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VariantHeads(2, 0, 2, DrawsFor("x")));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VariantHeads(2, 2, 0, DrawsFor("x")));
    }

    [Fact]
    public void VariantHeads_RefusesABatchNotLaidOutByItsVariants_OrOfAnotherWidth()
    {
        var heads = new VariantHeads(2, 2, 1, DrawsFor("0.heads"));

        Assert.Throws<ArgumentException>(() => heads.Forward(Tensor.From(new Shape(4, 2), new float[8]), Pass.Evaluation(new CpuBackend())));
        Assert.Throws<ArgumentException>(() => heads.Forward(Tensor.From(new Shape(2, 3, 2), new float[12]), Pass.Evaluation(new CpuBackend())));
        Assert.Throws<ArgumentException>(() => heads.Forward(Tensor.From(new Shape(2, 2, 3), new float[12]), Pass.Evaluation(new CpuBackend())));
    }

    [Fact]
    public void ABatchEnsembleAdapter_ThenVariantHeads_AnswersTheSameAsEachAloneComposed()
    {
        var adapter = new BatchEnsemble(2, 2, DrawsFor("0.adapter"));
        var heads = new VariantHeads(2, 2, 1, DrawsFor("1.heads"));

        adapter.Adapter.Replace(Tensor.From(new Shape(2, 2), [1f, 1f, 1f, 1f]));
        heads.Weight.Replace(Tensor.From(new Shape(4, 1), [1f, 1f, 1f, 1f]));
        heads.Bias.Replace(Tensor.From(new Shape(2, 1), [0f, 0f]));

        var input = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);
        var backend = new CpuBackend();
        var afterAdapter = adapter.Forward(input, Pass.Evaluation(backend));
        var output = heads.Forward(afterAdapter, Pass.Evaluation(backend));

        // Every variant's adapter row and head weight is 1, so both variants of a row answer the same: the row's own sum.
        Assert.Equal(new Shape(2, 2, 1), output.Shape);
        Assert.Equal([3f, 3f, 7f, 7f], output.Values.ToArray());
    }

    [Fact]
    public void OfVariants_IsTheMeanOverTheBatchOfTheMeanOverItsVariantsOfTheLoss_NotTheLossOfTheirMean()
    {
        var outputs = Tensor.From(new Shape(2, 2, 1), [2f, 4f, 6f, 8f]);
        var answers = Tensor.From(new Shape(2, 1), [3f, 5f]);

        var loss = new MeanSquaredError().OfVariants(outputs, answers, new CpuBackend());

        // (2-3)^2=1, (4-3)^2=1, (6-5)^2=1, (8-5)^2=9: mean of the four is 3.
        Assert.Equal(3f, loss.Values[0], 0.0001);
    }

    [Fact]
    public void VariantPredictions_BringsEachVariantsRawOutputThroughTheLossesActivationFirst_ThenTheirMean()
    {
        var outputs = Tensor.From(new Shape(2, 2, 1), [2f, 4f, 6f, 8f]);

        var predicted = new MeanSquaredError().VariantPredictions(outputs, new CpuBackend());

        // Identity activation: the mean of each row's two raw outputs, (2+4)/2=3 and (6+8)/2=7.
        Assert.Equal(new Shape(2, 1), predicted.Shape);
        Assert.Equal([3f, 7f], predicted.Values.ToArray());
    }

    [Fact]
    public void ADeclaredNetworkOfBatchEnsembleThenVariantHeads_IsLoweredWithTheVariantsAxisThroughout_AndIsDetectedByItsVariants()
    {
        var network = new Sequential().BatchEnsemble(3).Dense(4).Relu().VariantHeads(1).Lower(new Shape(2), new RandomStream(1));
        var backend = new CpuBackend();
        var batch = Tensor.From(new Shape(5, 2), [.. Enumerable.Range(0, 10).Select(at => (float)at)]);

        var raw = network.Forward(batch, Pass.Evaluation(backend));

        Assert.Equal(new Shape(5, 3, 1), raw.Shape);
        Assert.Equal(3, network.Variants);
    }

    [Fact]
    public void APlainDeclaredNetwork_HasNoVariants()
    {
        var network = new Sequential().Dense(3).Relu().Dense(1).Lower(new Shape(2), new RandomStream(1));

        Assert.Null(network.Variants);
    }

    [Fact]
    public void ANetworkOfBatchEnsembleAndVariantHeads_FitsAndPredictsOneRowPerExample_ThroughTheOrdinaryCompiledDoor()
    {
        var network = new Sequential().BatchEnsemble(4).Dense(6).Relu().VariantHeads(1).Lower(new Shape(2), new RandomStream(1));
        var backend = new CpuBackend();
        var features = Tensor.From(new Shape(8, 2), [.. Enumerable.Range(0, 16).Select(at => (float)at / 16)]);
        var answers = Tensor.From(new Shape(8, 1), [.. Enumerable.Range(0, 8).Select(at => (float)at / 8)]);
        var compiled = network.Compile(new Sgd(0.01), new MeanSquaredError());

        var history = compiled.Fit(new TrainingData(features, answers), null, new FitOptions(1) { Epochs = 2, Backend = backend });
        var predicted = compiled.Predict(features, backend);

        Assert.All(history.Epochs, epoch => Assert.True(double.IsFinite(epoch.Loss)));
        Assert.Equal(new Shape(8, 1), predicted.Shape);
        Assert.All(predicted.Values.ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void ABatchNormAfterAnOpenBatchEnsemble_IsRefused_ForItWouldMixStatisticsAcrossVariants()
    {
        var network = new Sequential().BatchEnsemble(3).Dense(4).BatchNorm().Relu().VariantHeads(1);

        var thrown = Assert.Throws<ArgumentException>(() => network.Lower(new Shape(2), new RandomStream(1)));
        Assert.Contains("BatchEnsemble opened", thrown.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AttentionAfterAnOpenBatchEnsemble_IsRefused_ForTheVariantsAreNotASeriesToAttendOver()
    {
        var network = new Sequential().BatchEnsemble(3).Attention(2);

        var thrown = Assert.Throws<ArgumentException>(() => network.Lower(new Shape(2), new RandomStream(1)));
        Assert.Contains("BatchEnsemble opened", thrown.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SinePositionsAfterAnOpenBatchEnsemble_IsRefused_ForTheVariantsAreNotPositionsInASeries()
    {
        var network = new Sequential().BatchEnsemble(3).SinePositions();

        var thrown = Assert.Throws<ArgumentException>(() => network.Lower(new Shape(2), new RandomStream(1)));
        Assert.Contains("BatchEnsemble opened", thrown.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnLstmAfterAnOpenBatchEnsemble_IsRefused_ForTheVariantsAreNotStepsOfATimeSeries()
    {
        var network = new Sequential().BatchEnsemble(3).Lstm(4);

        var thrown = Assert.Throws<ArgumentException>(() => network.Lower(new Shape(2), new RandomStream(1)));
        Assert.Contains("BatchEnsemble opened", thrown.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void LayerNormAndRmsNormAfterAnOpenBatchEnsemble_AreNotRefused_ForNeitherMixesStatisticsAcrossRows()
    {
        var layerNormed = new Sequential().BatchEnsemble(3).Dense(4).LayerNorm().VariantHeads(1).Lower(new Shape(2), new RandomStream(1));
        var rmsNormed = new Sequential().BatchEnsemble(3).Dense(4).RmsNorm().VariantHeads(1).Lower(new Shape(2), new RandomStream(1));

        Assert.NotNull(layerNormed);
        Assert.NotNull(rmsNormed);
    }
}
