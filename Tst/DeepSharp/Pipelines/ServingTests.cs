// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The handover serves as well as trains. A row that arrives a year from now is replayed through the saved
/// pipeline and handed over in exactly the shape the training rows were — the same columns in the same order,
/// refused for the same reasons — without an answer, since the answer is what is being asked. And because a
/// replay can drop rows, each served row says which of the handed-in rows it is.
/// </summary>
public class ServingTests
{
    private static PreparedData Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("age")
            .EncodeCategories()
            .Target("survived")
            .Build()
            .Run();

    [Fact]
    public void ServedRows_ComeOutInTheShapeTheTrainingRowsDid_WithoutAnAnswer()
    {
        var trained = Passengers();

        var served = trained.Served(new InMemoryRowSource(
            ["pclass", "sex", "age"], [["3", "female", "30"], ["1", "male", null]]));

        Assert.Equal(trained.Batch(Part.Train).FeatureNames, served.FeatureNames);
        Assert.Equal(2, served.RowCount);
        Assert.Equal(served.FeatureNames.Count, served.Features[0].Length);
        Assert.Equal([0, 1], served.HandedInAt);
    }

    [Fact]
    public void AServedRow_SaysWhichHandedInRowItIs_AfterTheWarmUpIsDropped()
    {
        IReadOnlyList<IReadOnlyList<string?>> Days(int count) =>
        [
            .. Enumerable.Range(0, count).Select(day => (IReadOnlyList<string?>)
            [
                new DateTime(2026, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                (100 + (day * 3 % 11)).ToString(CultureInfo.InvariantCulture),
            ]),
        ];

        var trained = Pdd.Create()
            .Read(new InMemoryRowSource(["Date", "close"], Days(60)), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("close"))
            .OrderBy("Date")
            .AddIndicator("sma5", Indicator.Sma, ["close"], 5)
            .DropWarmUp()
            .SplitByTime("Date", 0.70)
            .Normalise("close")
            .Drop("Date")
            .Build()
            .Run();

        var served = trained.Served(new InMemoryRowSource(["Date", "close"], Days(10)));

        Assert.Equal([4, 5, 6, 7, 8, 9], served.HandedInAt);
    }

    // Days of prices, oldest first, with an answer beside each price — or without one, as a served row comes.
    private static InMemoryRowSource Bars(int count, Answer answer)
    {
        string[] names = answer == Answer.Absent ? ["Date", "close"] : ["Date", "close", "up"];

        return new InMemoryRowSource(names, [.. Enumerable.Range(0, count).Select(day => (IReadOnlyList<string?>)
        [
            new DateTime(2026, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            (100 + (day * 3 % 11)).ToString(CultureInfo.InvariantCulture),
            .. answer switch
            {
                Answer.Known => [(day % 2).ToString(CultureInfo.InvariantCulture)],
                Answer.Empty => [(string?)null],
                _ => Array.Empty<string?>(),
            },
        ])]);
    }

    [Fact]
    public void AServedRowWithoutItsAnswer_IsServedThroughAWarmUpDrop()
    {
        // The answer a served row lacks arrives as a gap in every row, and a warm-up drop counted it as a column
        // that could not speak yet: every served row was refused.
        var trained = Pdd.Create()
            .Read(Bars(60, Answer.Known), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("close").Integer("up"))
            .OrderBy("Date")
            .AddIndicator("sma5", Indicator.Sma, ["close"], 5)
            .DropWarmUp()
            .SplitByTime("Date", 0.70)
            .Normalise("close")
            .Drop("Date")
            .Target("up")
            .Build()
            .Run();

        var served = trained.Served(Bars(10, Answer.Absent));

        Assert.Equal([4, 5, 6, 7, 8, 9], served.HandedInAt);
    }

    [Theory]
    [InlineData(Answer.Absent)]
    [InlineData(Answer.Empty)]
    public void AServedRow_IsNeverDroppedForTheAnswerItAwaits(Answer answer)
    {
        // Dropping the rows without an answer is how training rows without one are left out — and a served row
        // is exactly a row without one. It used to leave every served row out, and say nothing.
        var trained = Pdd.Create()
            .Read(Bars(60, Answer.Known), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("close").Integer("up"))
            .DropGaps("up")
            .SplitByTime("Date", 0.70)
            .Normalise("close")
            .Drop("Date")
            .Target("up")
            .Build()
            .Run();

        var served = trained.Served(Bars(10, answer));

        Assert.Equal(10, served.RowCount);
    }

    [Fact]
    public void ATrainingRowWithoutItsAnswer_IsStillDroppedByAGapDropOnTheAnswer()
    {
        // The fit still judges the answer: a training row without one cannot be learned from.
        var rows = Bars(60, Answer.Known);
        var gapped = new InMemoryRowSource(rows.ColumnNames, [.. rows.Rows.Select((row, day) => day == 7 ? (IReadOnlyList<string?>)[row[0], row[1], null] : row)]);

        var trained = Pdd.Create()
            .Read(gapped, "sixty days, one without its answer")
            .Declare(schema => schema.Timestamp("Date").Number("close").Integer("up"))
            .DropGaps("up")
            .SplitByTime("Date", 0.70)
            .Target("up")
            .Build()
            .Run();

        Assert.Equal(59, trained.Table.RowCount);
    }

    /// <summary>What a row holds of its answer.</summary>
    public enum Answer
    {
        /// <summary>The answer is there.</summary>
        Known,

        /// <summary>The column is there, and empty.</summary>
        Empty,

        /// <summary>The column is not there at all.</summary>
        Absent,
    }

    [Fact]
    public void AServedRow_AwaitsEveryAnswerItsOutputNames()
    {
        // A served row is the question, so every answer is what it lacks — not only the first one: a row without
        // the second answer used to be refused at the door for a column it cannot have.
        var trained = new Pipeline(
            new PipelineDeclaration(
            [
                new ReadRowsStep("three numbers"),
                new DeclareStep([.. new[] { "a", "b", "c" }.Select(name => new ColumnDeclaration(name, ColumnKind.Number, Optional: false))]),
                new SplitAtRandomStep(new SplitShares(0.50, 0, 0.50), 1),
                new NamesTheseAnswers("b", "c"),
            ]),
            new InMemoryRowSource(["a", "b", "c"], [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["10", "11", "12"]])).Run();

        var served = trained.Served(new InMemoryRowSource(["a"], [["13"]]));

        Assert.Equal(["a"], served.FeatureNames);
        Assert.Equal([0], served.HandedInAt);
    }

    [Fact]
    public void AServedRowWithAGap_IsRefusedAsATrainingRowWouldBe()
    {
        var trained = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .Normalise("age")
            .Target("survived")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(
            () => trained.Served(new InMemoryRowSource(["age"], [["30"], [null]])));

        Assert.Contains("Row 2 of 'age' is still a gap", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AServedRowStillHoldingWords_IsRefused()
    {
        var trained = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Text("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Build()
            .Run();

        Assert.Throws<InvalidOperationException>(
            () => trained.Served(new InMemoryRowSource(["sex"], [["female"]])));
        Assert.Throws<ArgumentNullException>(() => Handover.Served(null!, new InMemoryRowSource(["sex"], [])));
        Assert.Throws<ArgumentNullException>(() => trained.Served(null!));
    }
}
