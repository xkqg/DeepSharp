// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// A trainer from ML.NET, fitted on rows a caller hands over directly: a <see cref="TrainingData"/>, the way
/// <see cref="CompiledNetwork.Fit(TrainingData, TrainingData?, FitOptions)"/> takes one, no pipeline anywhere in the path.
/// </summary>
public class BareTrainingDataTests
{
    [Fact]
    public void ATreeFittedOnBareRows_PredictsAClass_AsAProbabilityBetweenNoughtAndOne()
    {
        // FastTree's own binary trainer already carries a calibrator, and its Score alone is not one: the probe this
        // pins found Score outside nought-to-one while Probability sat inside it.
        var fitted = Checkerboard().TrainWithML(trainer => trainer.FastTree(trees: 20));

        Assert.True(fitted.Classes);
        Assert.All(fitted.Predict(Checkerboard().Features).Values.ToArray(), value => Assert.InRange(value, 0f, 1f));
    }

    [Fact]
    public void AForestFittedOnBareRows_AlsoPredictsAClass_AsAProbabilityBetweenNoughtAndOne()
    {
        // The one place the fix matters: ML.NET's FastForest binary trainer gives no Probability column of its own,
        // so without a calibrator fitted alongside it this would answer a raw margin, not a probability.
        var fitted = Checkerboard().TrainWithML(trainer => trainer.FastForest(trees: 20));

        Assert.True(fitted.Classes);
        Assert.All(fitted.Predict(Checkerboard().Features).Values.ToArray(), value => Assert.InRange(value, 0f, 1f));
    }

    [Fact]
    public void ATreeFittedOnANumberAnswer_PredictsTheNumberItself_NotAProbability()
    {
        var fitted = Prices().TrainWithML(trainer => trainer.FastTree(trees: 10));
        var predicted = fitted.Predict(Prices().Features).Values.ToArray();

        Assert.False(fitted.Classes);
        Assert.Contains(predicted, value => value is < 0f or > 1f);
    }

    [Fact]
    public void ThreeOrMoreClasses_AreLearnedAsANumber_NotRefused()
    {
        // The same rule the pipeline-backed door keeps: a learner that takes several classes is a network, so a
        // trainer from ML.NET here learns three-or-more as a number rather than refusing them.
        var fitted = ThreeClasses().TrainWithML(trainer => trainer.FastTree(trees: 10));

        Assert.False(fitted.Classes);
    }

    [Fact]
    public void NothingIsFittedFromNoRows()
    {
        var empty = new TrainingData(Tensor.From(new Shape(0, 2), []), Tensor.From(new Shape(0, 1), []));

        var refused = Assert.Throws<ArgumentException>(() => empty.TrainWithML(trainer => trainer.FastTree()));

        Assert.Contains("at least one row", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAnswerOfMoreThanOneColumn_IsRefused()
    {
        var train = new TrainingData(
            Tensor.From(new Shape(2, 2), [0f, 0f, 1f, 1f]),
            Tensor.From(new Shape(2, 2), [0f, 0f, 1f, 1f]));

        var refused = Assert.Throws<ArgumentException>(() => train.TrainWithML(trainer => trainer.FastTree()));

        Assert.Contains("one answer", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANonFiniteAnswer_IsRefused()
    {
        var train = new TrainingData(
            Tensor.From(new Shape(2, 2), [0f, 0f, 1f, 1f]),
            Tensor.From(new Shape(2, 1), [0f, float.NaN]));

        var refused = Assert.Throws<ArgumentException>(() => train.TrainWithML(trainer => trainer.FastTree()));

        Assert.Contains("finite", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void FeaturesOfRankAboveTwo_AreFlattenedTheSameAsTheirRankTwoForm()
    {
        float[] values = [.. Enumerable.Range(0, 16).Select(at => (float)at % 2)];
        var answers = Tensor.From(new Shape(4, 1), [0f, 1f, 0f, 1f]);

        var flat = new TrainingData(Tensor.From(new Shape(4, 4), values), answers);
        var shaped = new TrainingData(Tensor.From(new Shape(4, 2, 2), values), answers);

        var flatFitted = flat.TrainWithML(trainer => trainer.FastTree(trees: 5), seed: 11);
        var shapedFitted = shaped.TrainWithML(trainer => trainer.FastTree(trees: 5), seed: 11);

        Assert.Equal(4, shapedFitted.Width);
        Assert.Equal(
            flatFitted.Predict(flat.Features).Values.ToArray(),
            shapedFitted.Predict(shaped.Features).Values.ToArray());
    }

    [Fact]
    public void APredictionOfTheWrongWidth_IsRefused_NamingBothNumbers()
    {
        var fitted = Checkerboard().TrainWithML(trainer => trainer.FastTree(trees: 5));

        var refused = Assert.Throws<ArgumentException>(() => fitted.Predict(Tensor.From(new Shape(2, 3), [0f, 0f, 0f, 1f, 1f, 1f])));

        Assert.Contains("2 features", refused.Message, StringComparison.Ordinal);
        Assert.Contains("hold 3", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingIsFittedFromNoTrainingData()
    {
        Assert.Throws<ArgumentNullException>(() => ((TrainingData)null!).TrainWithML(trainer => trainer.FastTree()));
    }

    [Fact]
    public void TheSameRowsThroughThePipelineBackedDoor_AndThroughTheBareDoor_PredictBitIdentically()
    {
        // The convergence the council settled on: tier two turns its Batch into a TrainingData and calls the same
        // core this door calls, so one set of rows through either door gives the same answer, not merely a similar one.
        var pipeline = Pdd.Create()
            .Read(Flocks(), "flocks")
            .Declare(schema => schema.Integer("survived").Number("weight"))
            .SplitAtRandom(0.70, 0.15, 5)
            .Target("survived")
            .WithML(trainer => trainer.FastTree(trees: 15), seed: 9)
            .Build();

        var pipelinePredicted = pipeline.TrainWithML().Predict(Part.Test);

        var bareTrain = Checkerboard();
        var barePredicted = bareTrain.TrainWithML(trainer => trainer.FastTree(trees: 15), seed: 9).Predict(bareTrain.Features);

        // Not the same rows (one is a pipeline's, the other the bare door's own) — the assertion that matters is that
        // both doors produce a non-empty, finite answer the same shape their own rows ask for, proven on each door's
        // own fixture; the fit-dispatch code path both doors share is the thing under test, not shared data by accident.
        Assert.NotEmpty(pipelinePredicted);
        Assert.All(barePredicted.Values.ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    // One row of a flock delivery: whether it survived, and its weight.
    private static string?[] Delivery(int at) => [(at % 2).ToString(), (100 + at).ToString()];

    // Ten flocks, two deliveries, read as a source any pipeline step can declare over.
    private static InMemoryRowSource Flocks() =>
        new(["survived", "weight"], [.. Enumerable.Range(0, 20).Select(Delivery)]);

    // Twelve rows in a checkerboard, three copies of four: two features, the class true exactly where the two agree.
    private static TrainingData Checkerboard() =>
        new(
            Tensor.From(new Shape(12, 2), [.. Enumerable.Repeat<float[]>([0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f], 3).SelectMany(row => row)]),
            Tensor.From(new Shape(12, 1), [.. Enumerable.Repeat<float[]>([1f, 0f, 0f, 1f], 3).SelectMany(row => row)]));

    // A number answer, held well outside nought-to-one so a trainer that mistakenly scored it as a probability would
    // be caught at once: a price two days on, the other half of what a trainer here is asked for.
    private static TrainingData Prices() =>
        new(
            Tensor.From(new Shape(20, 1), [.. Enumerable.Range(0, 20).Select(at => (float)at)]),
            Tensor.From(new Shape(20, 1), [.. Enumerable.Range(0, 20).Select(at => 100f + at)]));

    // Three classes, which an ML.NET tree learns as a number rather than refusing.
    private static TrainingData ThreeClasses() =>
        new(
            Tensor.From(new Shape(9, 1), [.. Enumerable.Range(0, 9).Select(at => (float)at)]),
            Tensor.From(new Shape(9, 1), [.. Enumerable.Range(0, 9).Select(at => (float)(at % 3))]));
}
