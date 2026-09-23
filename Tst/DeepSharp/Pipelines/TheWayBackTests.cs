// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Scale what a model is asked to predict and its predictions come back scaled. An error of 0.03 means
/// nothing until it is 0.03 of something, and a report in scaled units flatters every model equally — so
/// the way back belongs to the saved pipeline, and it is checked against values whose answer is already
/// known rather than assumed to work.
/// </summary>
public class TheWayBackTests
{
    private static string Titanic => Path.Join(RepoRoot(), "Samples", "data", "titanic.csv");

    private static PreparedData Fares(Action<FittingBuilder> how)
    {
        var fitting = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Number("fare").Integer("pclass"))
            .SplitAtRandom(0.70, 0.15, 0.15, seed: 4);

        how(fitting);

        return fitting.Target("fare").Build().Run();
    }

    [Theory]
    [InlineData(Scale.Standard)]
    [InlineData(Scale.MinMax)]
    [InlineData(Scale.MaxAbs)]
    [InlineData(Scale.Robust)]
    [InlineData(Scale.Quantile)]
    [InlineData(Scale.Power)]
    public void EveryScaleCanBeUndone(Scale scale)
    {
        var prepared = Fares(fitting => fitting.Normalise("fare", scale));
        var scaled = (Column<double>)prepared.Table["fare"];

        // Row 1 of the file is a fare of 7.25. Whatever the scaling did to it, asking for it back has to
        // give 7.25 again -- and the run already checked this for every row before handing anything over.
        var back = prepared.BackToOriginal(scaled[0]!.Value);

        Assert.Equal(7.25, back, 4);
    }

    [Theory]
    [InlineData(Maths.Log)]
    [InlineData(Maths.Log1P)]
    [InlineData(Maths.Sqrt)]
    [InlineData(Maths.Reciprocal)]
    public void EveryReshapingCanBeUndoneToo(Maths maths)
    {
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Number("fare").Integer("pclass"))
            // A fare of nought is a free passage and there are 15 of them, so a logarithm of the fare
            // itself does not exist. Shifted by the class it becomes a number every reshaping can take.
            .AddFeature("paid", "fare", Arithmetic.Plus, "pclass")
            .Reshape("paid", maths)
            .SplitAtRandom(0.70, 0.15, 0.15, seed: 4)
            .Target("paid")
            .Build();

        var run = prepared.Run();
        var shaped = (Column<double>)run.Table["paid"];

        Assert.Equal(10.25, run.BackToOriginal(shaped[0]!.Value), 4);
    }

    [Fact]
    public void AReshapingAndAScalingTogether_ComeBackInThatOrder()
    {
        // The two steps are undone backwards: first the scaling, then the logarithm. Done the other way
        // round the number that comes out is in no units at all.
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Number("fare").Integer("pclass"))
            .AddFeature("paid", "fare", Arithmetic.Plus, "pclass")
            .Reshape("paid", Maths.Log1P)
            .SplitAtRandom(0.70, 0.15, 0.15, seed: 4)
            .Normalise("paid", Scale.Standard)
            .Target("paid")
            .Build()
            .Run();

        var value = ((Column<double>)prepared.Table["paid"])[0]!.Value;

        Assert.Equal(10.25, prepared.BackToOriginal(value), 4);
    }

    [Fact]
    public void ManyPredictionsComeBackAtOnce()
    {
        var prepared = Fares(fitting => fitting.Normalise("fare", Scale.Standard));
        var scaled = (Column<double>)prepared.Table["fare"];

        var back = prepared.BackToOriginal(
            Enumerable.Range(0, 5).Select(row => scaled[row]!.Value));

        Assert.Equal([7.25, 71.2833, 7.925, 53.1, 8.05], back.Select(value => Math.Round(value, 4)));
    }

    [Fact]
    public void ATargetNobodyTouched_ComesBackAsItself()
    {
        var prepared = Fares(_ => { });

        Assert.Equal(7.25, prepared.BackToOriginal(7.25), 6);
    }

    [Fact]
    public void APipelineWithNoTargetHasNoUnitsToComeBackTo()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Number("fare"))
            .SplitAtRandom(0.70, 0.15, 0.15)
            .Normalise("fare")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.BackToOriginal(0.5));

        Assert.Contains("names no target", refused.Message);
        Assert.Throws<ArgumentNullException>(() => prepared.BackToOriginal(null!));
    }

    [Theory]
    [InlineData(Maths.Abs)]
    [InlineData(Maths.Sign)]
    public void AReshapingThatThrewSomethingAway_SaysSoRatherThanGuessing(Maths maths)
    {
        // Losing the sign cannot be undone, and a number that came back wrong would be worse than none.
        var step = new MathsStep("fare", maths);

        var refused = Assert.Throws<InvalidOperationException>(() => step.Undo(1, null));

        Assert.Contains("cannot be put back", refused.Message);
    }

    [Fact]
    public void AndTheRunSaysSoBeforeAnythingIsHandedOver()
    {
        // The check scikit-learn calls check_inverse: transform, then undo, and compare against what was
        // read. A pipeline whose way back does not lead back is stopped here rather than at the point
        // somebody reads a report in units nobody can name.
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Number("fare"))
                .Reshape("fare", Maths.Sign)
                .SplitAtRandom(0.70, 0.15, 0.15)
                .Target("fare")
                .Build()
                .Run());

        Assert.Contains("cannot be put back", refused.Message);
    }

    [Fact]
    public void ASquareRootOfANegativePrediction_IsRefusedRatherThanImagined()
    {
        var step = new MathsStep("fare", Maths.Square);

        Assert.Throws<InvalidOperationException>(() => step.Undo(-1, null));
        Assert.Throws<InvalidOperationException>(() => new MathsStep("fare", Maths.Reciprocal).Undo(0, null));
        Assert.Throws<ArgumentNullException>(() => new NormaliseStep("fare").Undo(1, null));
    }

    [Fact]
    public void ATargetOfWordsIsNotSomethingToComeBackFrom()
    {
        // Predicting a category is an ordinary thing; there is simply no arithmetic to reverse, and the
        // check knows the difference between that and a broken way back.
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Category("sex").Number("fare"))
            .SplitAtRandom(0.70, 0.15, 0.15)
            .Target("sex")
            .Build()
            .Run();

        Assert.Equal(891, prepared.Table.RowCount);
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "DeepSharp.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return directory!.FullName;
    }
}
