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
    private static string Titanic => Repository.Data("titanic.csv");

    private static PreparedData Fares(Action<FittingBuilder> how)
    {
        var fitting = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Number("fare").Integer("pclass"))
            .SplitAtRandom(0.70, 0.15, seed: 4);

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
            .SplitAtRandom(0.70, 0.15, seed: 4)
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
            .SplitAtRandom(0.70, 0.15, seed: 4)
            .Normalise("paid", Scale.Standard)
            .Target("paid")
            .Build()
            .Run();

        var value = ((Column<double>)prepared.Table["paid"])[0]!.Value;

        Assert.Equal(10.25, prepared.BackToOriginal(value), 4);
    }

    [Fact]
    public void AnAnswerMadeFromAScaledColumn_ComesBackThroughTheScalingToo()
    {
        // A reshaping into a new column leaves the column it read behind, and the way back goes on there: undoing
        // the logarithm alone came back in the scaled units, and the run could not tell, since the answer it would
        // have compared with was never read.
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Number("fare"))
            .SplitAtRandom(0.70, 0.15, seed: 4)
            .Normalise("fare", Scale.MinMax)
            .Add(new MathsStep("fare", Maths.Log1P, "shaped"))
            .Drop("fare")
            .Target("shaped")
            .Build()
            .Run();

        var shaped = ((Column<double>)prepared.Table["shaped"])[0]!.Value;

        Assert.Equal(7.25, prepared.BackToOriginal(shaped), 4);
    }

    [Fact]
    public void ABrokenWayBackThroughAColumnMadeFromAnother_IsRefusedByTheRun()
    {
        // The run compares the way back with what was read, and for an answer made from another column that is
        // the other column as read. The check looked for the answer among the columns read, found nothing, and
        // let a way back through that doubles every fare.
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Number("fare"))
                .Add(new Doubled("fare"))
                .Reshape("fare", Maths.Log1P, "shaped")
                .SplitAtRandom(0.70, 0.15)
                .Drop("fare")
                .Target("shaped")
                .Build()
                .Run());

        Assert.Contains("The way back for 'shaped' does not lead back", refused.Message, StringComparison.Ordinal);
        Assert.Contains("row 1 was 7.25", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAnswerThatIsAMomentInTime_IsNotSomethingToComeBackFrom()
    {
        // The check read every answer that was not words as numbers, and a moment in time is neither.
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["Date", "close"], [["2026-01-01", "1"], ["2026-01-02", "2"], ["2026-01-03", "3"], ["2026-01-04", "4"]]), "four days")
            .Declare(schema => schema.Timestamp("Date").Number("close"))
            .SplitAtRandom(0.50, seed: 1)
            .Target("Date")
            .Build()
            .Run();

        Assert.Equal(4, prepared.Table.RowCount);
    }

    // As read: four flocks, how many of something each counted and how many birds it had.
    private static readonly double[] Counts = [100, 300, 80, 90];

    // The count is made a share per bird, so its way back needs the birds of its own row.
    private static PreparedData PerBird(string? undoBy = null) =>
        Pdd.Create()
            .Read(new InMemoryRowSource(["count", "chicks"], [["100", "50"], ["300", "100"], ["80", "40"], ["90", "30"]]), "four flocks")
            .Declare(schema => schema.Number("count", "chicks"))
            .SplitAtRandom(0.50, seed: 1)
            .Add(new Per("count", "chicks", undoBy))
            .Target("count")
            .Build()
            .Run();

    [Fact]
    public void AWayBackThatNeedsTheRow_IsNotTakenWithoutIt()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => PerBird().BackToOriginal(2.0));

        Assert.Contains("only with the row", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictionsForAPart_ComeBackWithTheRowsTheyWereMadeFor()
    {
        // A share per bird comes back as a count only by the birds of its own row: the number alone cannot say.
        var prepared = PerBird();
        var batch = prepared.Batch(Part.Test);

        var back = prepared.BackToOriginal(batch.Answers!, Part.Test);

        Assert.Equal(
            prepared.Table.Identities.Where((_, row) => prepared.Parts[row] == Part.Test).Select(identity => Counts[identity.ReadAt]),
            back.Select(answers => answers[0]));
    }

    [Fact]
    public void AWayBackThatReadsTheWrongColumnOfTheRow_IsRefusedByTheRun()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => PerBird(undoBy: "count"));

        Assert.Contains("The way back for 'count' does not lead back", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWayBackThatReadsAColumnItNeverNamed_IsToldTheColumnIsNotKept()
    {
        // A step reads from the row as it was read only what it names: that is all that is kept of it.
        var refused = Assert.Throws<InvalidOperationException>(() => PerBird(undoBy: "weight"));

        Assert.Contains("'weight'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("not kept", refused.Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => default(RowAsRead)["count"]);
    }

    [Fact]
    public void EveryAnswerOfAnOutput_ComesBackInItsOwnUnits()
    {
        var prepared = new Pipeline(
            new PipelineDeclaration(
            [
                new ReadRowsStep("three numbers"),
                new DeclareStep([.. new[] { "a", "b", "c" }.Select(name => new ColumnDeclaration(name, ColumnKind.Number, Optional: false))]),
                new SplitAtRandomStep(new SplitShares(0.50, 0, 0.50), 1),
                new NormaliseStep("b"),
                new NamesTheseAnswers("b", "c"),
            ]),
            new InMemoryRowSource(["a", "b", "c"], [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["10", "11", "12"]])).Run();

        var batch = prepared.Batch(Part.Train);
        var back = prepared.BackToOriginal(batch.Answers!, Part.Train);

        // As read, every row's c is its b and one more; as handed over, b was scaled and c was not.
        Assert.All(back, answers => Assert.Equal(answers[0] + 1, answers[1], 6));
        Assert.All(batch.Answers!, answers => Assert.NotEqual(answers[0] + 1, answers[1], 6));
    }

    [Fact]
    public void PredictionsThatDoNotFitTheRows_AreRefused()
    {
        var prepared = PerBird();
        var rows = prepared.CountIn(Part.Test);

        Assert.Throws<ArgumentException>(() => prepared.BackToOriginal([.. Enumerable.Repeat(new[] { 1.0 }, rows + 1)], Part.Test));
        Assert.Throws<ArgumentException>(() => prepared.BackToOriginal([.. Enumerable.Repeat(new[] { 1.0, 2.0 }, rows)], Part.Test));
        Assert.Throws<ArgumentNullException>(() => prepared.BackToOriginal(null!, Part.Test));
    }

    [Fact]
    public void PredictionsForServedRows_ComeBackWithTheRowsThatWereHandedIn()
    {
        var prepared = PerBird();
        var handedIn = new InMemoryRowSource(["chicks"], [["50"], ["20"]]);
        var served = prepared.Served(handedIn);

        var back = prepared.BackToOriginal([[2.0], [4.0]], served, handedIn);

        Assert.Equal([100.0, 80.0], back.Select(answers => answers[0]));
    }

    [Fact]
    public void ServedRowsHandedInAgainInAnotherOrder_AreRefusedRatherThanAnsweredForTheWrongRow()
    {
        var prepared = PerBird();
        var served = prepared.Served(new InMemoryRowSource(["chicks"], [["50"], ["20"]]));

        var reordered = Assert.Throws<InvalidOperationException>(
            () => prepared.BackToOriginal([[2.0], [4.0]], served, new InMemoryRowSource(["chicks"], [["20"], ["50"]])));
        var shorter = Assert.Throws<InvalidOperationException>(
            () => prepared.BackToOriginal([[2.0], [4.0]], served, new InMemoryRowSource(["chicks"], [["50"]])));

        Assert.Contains("Served row 1", reordered.Message, StringComparison.Ordinal);
        Assert.Contains("Served row 2", shorter.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(
            () => prepared.BackToOriginal([[2.0], [4.0]], served with { Keys = null }, new InMemoryRowSource(["chicks"], [["50"], ["20"]])));
        Assert.Throws<ArgumentException>(
            () => prepared.BackToOriginal([[2.0], [4.0]], served with { Keys = [served.Keys![0]] }, new InMemoryRowSource(["chicks"], [["50"], ["20"]])));
        Assert.Throws<ArgumentNullException>(() => prepared.BackToOriginal([[2.0], [4.0]], served, null!));
    }

    [Fact]
    public void APipelineLoadedFromItsFile_PutsServedPredictionsBackToo()
    {
        var trained = Fares(fitting => fitting.Normalise("fare", Scale.Standard));
        var loaded = PreparedData.FromJson(trained.ToJson(), StepCatalog.BuiltIn());
        var handedIn = new InMemoryRowSource(["pclass"], [["3"], ["1"]]);

        var back = loaded.BackToOriginal([[0.0], [1.0]], loaded.Served(handedIn), handedIn);

        Assert.Equal(trained.BackToOriginal([0.0, 1.0]), back.Select(answers => answers[0]));
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
            .SplitAtRandom(0.70, 0.15)
            .Normalise("fare")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.BackToOriginal(0.5));
        var forAPart = Assert.Throws<InvalidOperationException>(() => prepared.BackToOriginal([], Part.Train));

        Assert.Contains("names no answer", refused.Message);
        Assert.Contains("names no answer", forAPart.Message);
        Assert.Throws<ArgumentNullException>(() => prepared.BackToOriginal(null!));
    }

    [Fact]
    public void AnOutputOfSeveralAnswers_IsNotPutBackOneNumberAtATime()
    {
        // One number goes back into the units of one column; which of several it belongs to is not something to
        // guess.
        var prepared = new Pipeline(
            new PipelineDeclaration(
            [
                new ReadRowsStep("three numbers"),
                new DeclareStep([.. new[] { "a", "b", "c" }.Select(name => new ColumnDeclaration(name, ColumnKind.Number, Optional: false))]),
                new SplitAtRandomStep(new SplitShares(0.50, 0, 0.50), 1),
                new NamesTheseAnswers("b", "c"),
            ]),
            new InMemoryRowSource(["a", "b", "c"], [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["10", "11", "12"]])).Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.BackToOriginal(0.5));

        Assert.Contains("2 answers", refused.Message, StringComparison.Ordinal);
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
                .SplitAtRandom(0.70, 0.15)
                .Target("fare")
                .Build()
                .Run());

        Assert.Contains("cannot be put back", refused.Message);
    }

    [Fact]
    public void AWayBackThatLeadsSomewhereElse_IsRefusedWithTheRowItWasFoundOn()
    {
        // A step that doubles the target and claims to undo itself by doing nothing: every row comes back
        // as twice what it was read as, and the run says so before anything is handed over.
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Number("fare"))
                .Add(new Doubled("fare"))
                .SplitAtRandom(0.70, 0.15)
                .Target("fare")
                .Build()
                .Run());

        Assert.Contains("does not lead back", refused.Message, StringComparison.Ordinal);
        Assert.Contains("row 1 was 7.25", refused.Message, StringComparison.Ordinal);
    }

    /// <summary>A step whose way back is wrong on purpose.</summary>
    private sealed class Doubled(string column) : IAddsColumns, IUndoesItself
    {
        public string Verb => "test.doubled";

        public string Produces => column;

        public void AddTo(Table table) =>
            table.Put(new Column<double>(column, ColumnKind.Number, table.NumbersOf(column).Select(value => value * 2)));

        public double Undo(double value, FittedStepValues? fitted) => value;

        public void WriteTo(System.Text.Json.Utf8JsonWriter writer) =>
            throw new NotSupportedException("A test step is never written down.");
    }

    /// <summary>A count made a share of the birds in its row, whose way back needs that row as it was read.</summary>
    private sealed class Per(string column, string by, string? undoBy) : IAddsColumns, IUndoesItself
    {
        public string Verb => "test.per";

        public string Produces => column;

        public IReadOnlyList<ColumnRead> ColumnsRead => [new(column, ColumnKinds.Numbers), new(by, ColumnKinds.Numbers)];

        public void AddTo(Table table)
        {
            var birds = table.NumbersOf(by);

            table.Put(new Column<double>(column, ColumnKind.Number, table.NumbersOf(column).Select((value, row) => value / birds[row])));
        }

        public double Undo(double value, FittedStepValues? fitted) =>
            throw new InvalidOperationException($"'{column}' comes back only with the row it was made from: it was divided by '{by}' there.");

        public double Undo(double value, FittedStepValues? fitted, RowAsRead row) => value * row[undoBy ?? by]!.Value;

        public void WriteTo(System.Text.Json.Utf8JsonWriter writer) =>
            throw new NotSupportedException("A test step is never written down.");
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
            .SplitAtRandom(0.70, 0.15)
            .Target("sex")
            .Build()
            .Run();

        Assert.Equal(891, prepared.Table.RowCount);
    }
}
