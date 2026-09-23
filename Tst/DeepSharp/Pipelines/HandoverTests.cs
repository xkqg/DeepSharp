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
    private static string Data(string file) => Path.Join(RepoRoot(), "Samples", "data", file);

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

    private static PreparedData Passengers() =>
        Pdd.Create()
            .ReadCsv(Data("titanic.csv"))
            .Declare(schema => schema
                .Integer("survived", "pclass", "sibsp")
                .Text("sex")
                .Number("fare")
                .Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15, 0.15)
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
        var batch = prepared.Batch(Split.Train);

        Assert.Equal(prepared.CountIn(Split.Train), batch.RowCount);
        Assert.Equal(batch.Width, batch.Features[0].Length);
        Assert.DoesNotContain("survived", batch.FeatureNames);
        Assert.Contains("age_was_missing", batch.FeatureNames);
        Assert.Contains("sex_female", batch.FeatureNames);
    }

    [Fact]
    public void TheAnswerIsHandedOverSeparately_NeverAmongTheNumbers()
    {
        var batch = Passengers().Batch(Split.Test);

        Assert.NotNull(batch.Labels);
        Assert.Equal(batch.RowCount, batch.Labels!.Count);
        Assert.All(batch.Labels, label => Assert.True(label is 0 or 1));
    }

    [Fact]
    public void EverySplitCanBeHandedOverOnItsOwn()
    {
        var prepared = Passengers();

        var counted = new[] { Split.Train, Split.Validation, Split.Test }
            .Sum(split => prepared.Batch(split).RowCount);

        Assert.Equal(891, counted);
    }

    [Fact]
    public void APipelineWithNoTargetHandsOverNoAnswers()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Data("titanic.csv"))
            .Declare(schema => schema.Integer("pclass"))
            .SplitAtRandom(0.70, 0.15, 0.15)
            .Build()
            .Run();

        Assert.Null(prepared.Batch(Split.Train).Labels);
    }

    [Fact]
    public void AColumnStillHoldingWords_IsRefusedAtTheHandover()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Data("titanic.csv"))
            .Declare(schema => schema.Text("sex").Integer("pclass"))
            .SplitAtRandom(0.70, 0.15, 0.15)
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Split.Train));

        Assert.Contains("sex", refused.Message);
    }

    [Fact]
    public void AGapThatWasNeverFilled_IsRefusedAtTheHandover()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Data("titanic.csv"))
            .Declare(schema => schema.Optional("age", ColumnKind.Number))
            .SplitAtRandom(0.70, 0.15, 0.15)
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Split.Train));

        Assert.Contains("age", refused.Message);
        Assert.Contains("still a gap", refused.Message);
    }

    [Fact]
    public void ATargetThatNeverReachedTheEnd_IsRefused()
    {
        // Encoding the target takes the column away and puts columns per category in its place, so the
        // pipeline is asked to predict something that is no longer there.
        var prepared = Pdd.Create()
            .ReadCsv(Data("titanic.csv"))
            .Declare(schema => schema.Text("sex").Integer("pclass"))
            .SplitAtRandom(0.70, 0.15, 0.15)
            .Encode("sex")
            .Target("sex")
            .Build()
            .Run();

        Assert.Throws<InvalidOperationException>(() => prepared.Batch(Split.Train));
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
        var loaded = PreparedData.FromJson(trained.ToJson());

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
    public void AFittedHalfWrittenUnderSomethingOtherThanAPosition_IsRefused()
    {
        const string json = """
            {"declaration":[{"step":"read.csv","path":"x.csv"}],"fitted":{"fill":{"value":1}}}
            """;

        Assert.Throws<FormatException>(() => PreparedData.FromJson(json));
    }

    [Fact]
    public void AFitThatClaimsToHaveLearnedAWord_IsRefused()
    {
        const string json = """
            {"declaration":[{"step":"read.csv","path":"x.csv"}],"fitted":{"0":{"value":"a lot"}}}
            """;

        Assert.Throws<FormatException>(() => PreparedData.FromJson(json));
    }

    [Fact]
    public void ASavedPipelineWithNothingLearnedYet_LoadsAnyway()
    {
        var declaration = Pdd.Create().ReadCsv("x.csv").Declaration;

        Assert.Empty(PreparedData.FromJson(declaration.ToJson()).Fitted);
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
