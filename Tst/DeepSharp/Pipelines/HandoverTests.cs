// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Where the pipeline stops and something that learns begins: rows of numbers, their column names in a
/// fixed order, and the answer handed over separately. And the other half of the same idea — the same
/// declaration replayed over rows nobody had seen, with the numbers the training rows produced.
/// </summary>
public class HandoverTests
{
    private static PreparedData Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema
                .Integer("survived", "pclass", "sibsp")
                .Text("sex")
                .Number("fare")
                .Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Encode("sex")
            .Normalise("age", "fare")
            .Target("survived")
            .Build()
            .Run();

    [Fact]
    public void AHandover_IsRowsOfNumbersWithTheirNamesInOrder()
    {
        var prepared = Passengers();
        var batch = prepared.Batch(Part.Train);

        Assert.Equal(prepared.CountIn(Part.Train), batch.RowCount);
        Assert.Equal(batch.Width, batch.Features[0].Length);
        Assert.DoesNotContain("survived", batch.FeatureNames);
        Assert.Contains("age_was_missing", batch.FeatureNames);
        Assert.Contains("sex_female", batch.FeatureNames);
    }

    [Fact]
    public void TheAnswerIsHandedOverSeparately_NeverAmongTheNumbers()
    {
        var batch = Passengers().Batch(Part.Test);

        Assert.NotNull(batch.Labels);
        Assert.Equal(batch.RowCount, batch.Labels!.Count);
        Assert.All(batch.Labels, label => Assert.True(label is 0 or 1));
    }

    [Fact]
    public void EverySplitCanBeHandedOverOnItsOwn()
    {
        var prepared = Passengers();

        var counted = new[] { Part.Train, Part.Validation, Part.Test }
            .Sum(split => prepared.Batch(split).RowCount);

        Assert.Equal(891, counted);
    }

    [Fact]
    public void APipelineWithNoTargetHandsOverNoAnswers()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("pclass"))
            .SplitAtRandom(0.70, 0.15)
            .Build()
            .Run();

        Assert.Null(prepared.Batch(Part.Train).Labels);
    }

    [Fact]
    public void AColumnStillHoldingWords_IsRefusedAtTheHandover()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Text("sex").Integer("pclass"))
            .SplitAtRandom(0.70, 0.15)
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Train));

        Assert.Contains("sex", refused.Message);
    }

    [Fact]
    public void AGapThatWasNeverFilled_IsRefusedAtTheHandover()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Optional("age", ColumnKind.Number))
            .SplitAtRandom(0.70, 0.15)
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Train));

        Assert.Contains("age", refused.Message);
        Assert.Contains("still a gap", refused.Message);
    }

    [Theory]
    [InlineData("Infinity")]
    [InlineData("-Infinity")]
    [InlineData("NaN")]
    public void ANumberThatIsNotFinite_IsRefusedAtTheHandover(string written)
    {
        // A gap was refused here and a not-a-number or an infinity walked straight through: the check was
        // for an absent value only. A model handed an infinity learns nothing and says nothing about it.
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText($"fare,survived\n7.25,0\n{written},1\n8.05,0\n"), "three passengers")
            .Declare(schema => schema.Number("fare").Integer("survived"))
            .SplitAtRandom(0.34, seed: 1)
            .Target("survived")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(
            () => new[] { Part.Train, Part.Test }.Select(part => prepared.Batch(part)).ToArray());

        Assert.Contains("Row 2 of 'fare'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("not a finite number", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAnswerThatIsNotFinite_IsRefusedToo()
    {
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("fare,price\n7.25,1\n8.05,Infinity\n"), "two rows")
            .Declare(schema => schema.Number("fare", "price"))
            .SplitAtRandom(0.50, seed: 1)
            .Target("price")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(
            () => new[] { Part.Train, Part.Test }.Select(part => prepared.Batch(part)).ToArray());

        Assert.Contains("'price'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATargetThatNeverReachedTheEnd_IsRefused()
    {
        // Encoding the target takes the column away and puts columns per category in its place, so the
        // pipeline is asked to predict something that is no longer there — refused where the target is
        // written, since the columns there are known.
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create()
                .ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Text("sex").Integer("pclass"))
                .SplitAtRandom(0.70, 0.15)
                .Encode("sex")
                .Target("sex"));

        Assert.Equal("target", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void ATargetThatNeverReachedTheEnd_IsRefusedAtTheHandover_WhenNoDeclarationCouldTell()
    {
        // A step from elsewhere that takes the answer away without saying so leaves nothing any declaration
        // could follow; the handover still says what is missing.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitAtRandom(0.70, 0.15)
            .Target("survived")
            .Add(new ForgetStep("survived"))
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Train));

        Assert.Contains("predicts 'survived'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANewRowIsPreparedWithTheNumbersTheTrainingRowsProduced()
    {
        var prepared = Passengers();
        var learnedFill = prepared.Fitted.Values.First(values => values.Numbers.ContainsKey("gaps")).Number("value");

        // One passenger, with no age at all, arriving long after the model was trained.
        var arriving = new InMemoryRowSource(
            ["survived", "pclass", "sibsp", "sex", "fare", "age"],
            [["0", "3", "0", "female", "7.75", null]]);

        var replayed = prepared.Replay(arriving);

        Assert.Equal(1, replayed.RowCount);
        Assert.Equal(1, ((Column<double>)replayed["age_was_missing"])[0]);

        // The age it was given is the median of the training rows, not of anything it has seen since.
        var scaled = ((Column<double>)replayed["age"])[0]!.Value;
        var centre = prepared.Fitted.Values.Last(values => values.Numbers.ContainsKey("centre")).Number("centre");
        var spread = prepared.Fitted.Values.Last(values => values.Numbers.ContainsKey("spread")).Number("spread");

        Assert.NotEqual(0, learnedFill);
        Assert.False(double.IsNaN(scaled));
        Assert.False(double.IsNaN(centre));
        Assert.False(double.IsNaN(spread));
    }

    [Fact]
    public void ReplayLearnsNothing_SoOneRowIsNotScaledAgainstItself()
    {
        var prepared = Passengers();

        var one = prepared.Replay(new InMemoryRowSource(
            ["survived", "pclass", "sibsp", "sex", "fare", "age"],
            [["1", "1", "0", "male", "100", "40"]]));

        var other = prepared.Replay(new InMemoryRowSource(
            ["survived", "pclass", "sibsp", "sex", "fare", "age"],
            [["1", "1", "0", "male", "100", "40"], ["0", "3", "0", "female", "5", "5"]]));

        // Scaled against the fit, not against the batch: the same row gets the same numbers whether it
        // arrives alone or in company. A step that re-learned here would give two different answers.
        Assert.Equal(((Column<double>)one["fare"])[0], ((Column<double>)other["fare"])[0]);
    }

    [Fact]
    public void ASavedPipelineIsLoadedBackAndPreparesANewRowTheSameWay()
    {
        // The whole promise of saving both halves: a host that has the file and no data at all can take a
        // row it has never seen and hand a model exactly the numbers that model was trained on.
        var trained = Passengers();
        var loaded = PreparedData.FromJson(trained.ToJson(), StepCatalog.BuiltIn());

        var arriving = new InMemoryRowSource(
            ["survived", "pclass", "sibsp", "sex", "fare", "age"],
            [["0", "3", "0", "female", "7.75", null]]);

        var fromTraining = trained.Replay(arriving);
        var fromFile = loaded.Replay(arriving);

        Assert.Equal(trained.Declaration, loaded.Declaration);
        Assert.Equal(0, loaded.Table.RowCount);
        Assert.Equal(
            fromTraining.Columns.Select(column => column.Name),
            fromFile.Columns.Select(column => column.Name));
        Assert.Equal(((Column<double>)fromTraining["age"])[0], ((Column<double>)fromFile["age"])[0]);
        Assert.Equal(((Column<double>)fromTraining["fare"])[0], ((Column<double>)fromFile["fare"])[0]);
    }

    [Fact]
    public void AFittedPipelineWithAStepFromAnotherPackage_SurvivesTheRoundTrip()
    {
        // A declaration could be read with a catalog that knows another package's verbs; the fitted
        // pipeline could not, so a pipeline with an indicator in it could be saved and never served.
        var trained = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .OrderBy("Date")
            .AddIndicator("sma5", Indicator.Sma, ["AAPL.Close"], 5)
            .DropWarmUp()
            .SplitByTime("Date", 0.70, 0.15)
            .Normalise("sma5")
            .Build()
            .Run();

        var refused = Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(trained.ToJson(), StepCatalog.BuiltIn()));

        Assert.Contains("DeepSharp.Pipelines.Indicators", refused.Message, StringComparison.Ordinal);

        var loaded = PreparedData.FromJson(trained.ToJson(), StepCatalog.BuiltIn().WithIndicators());

        Assert.Equal(trained.Declaration, loaded.Declaration);
        Assert.Equal(trained.Fitted.Keys, loaded.Fitted.Keys);
        Assert.Throws<ArgumentNullException>(() => PreparedData.FromJson(trained.ToJson(), null!));
    }

    [Fact]
    public void ASavedPipelineWithNothingLearnedYet_LoadsAnyway()
    {
        var declaration = Pdd.Create().ReadCsv("x.csv").Declaration;

        Assert.Empty(PreparedData.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()).Fitted);
    }

    [Fact]
    public void APipelineWithNoColumnsAtAll_HasNothingToReplay()
    {
        var prepared = new PreparedData(
            new PipelineDeclaration([new ReadCsvStep("x.csv")]),
            new Table([]),
            [],
            new Dictionary<int, FittedStepValues>());

        Assert.Throws<InvalidOperationException>(
            () => prepared.Replay(new InMemoryRowSource(["a"], [["1"]])));
    }
}
