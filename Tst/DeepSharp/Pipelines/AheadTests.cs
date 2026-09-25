// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// An answer read from later rows: the price five days on, or the return on today's price by then. It is made from
/// the rows in their declared order when the pipeline is fitted, it is what a served row asks for, and it comes back
/// as a price. Reading ahead is only for an output, and only across a split in time that keeps a gap at least as
/// wide as how far it reads — or the last training rows learn their answers from the rows a model is measured on.
/// </summary>
public class AheadTests
{
    private static double Close(int day) => 100 + (day * 3 % 11);

    private static IReadOnlyList<IReadOnlyList<string?>> Days(int count, int from = 0) =>
    [
        .. Enumerable.Range(from, count).Select(day => (IReadOnlyList<string?>)
        [
            new DateTime(2026, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            Close(day).ToString(CultureInfo.InvariantCulture),
        ]),
    ];

    private static FittingBuilder Split(int gap = 5, Remainder remainder = Remainder.Drop, params string[] orderBy) =>
        Pdd.Create()
            .Read(new InMemoryRowSource(["Date", "close"], Days(60)), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("close"), remainder)
            .OrderBy(orderBy.Length == 0 ? ["Date"] : orderBy)
            .SplitByTime("Date", 0.70, 0.15, gap)
            .Drop("Date");

    private static PreparedData Priced(AheadAs form, Remainder remainder = Remainder.Drop) =>
        Split(remainder: remainder).Ahead("close", 5, form).Build().Run();

    private static int[] DaysIn(PreparedData prepared, Part part) =>
        [.. prepared.Table.Identities.Where((_, row) => prepared.Parts[row] == part).Select(identity => identity.ReadAt)];

    [Fact]
    public void TheAnswerFiveDaysAhead_IsThePriceThen_InTheDeclaredOrder()
    {
        var prepared = Priced(AheadAs.Value);
        var answers = prepared.Table.NumbersOf("close.ahead5");

        for (var day = 0; day < 55; day++)
        {
            Assert.Equal(Close(day + 5), answers[day]);
        }

        Assert.All(answers.Skip(55), answer => Assert.Null(answer));
        Assert.Equal(["close.ahead5"], prepared.Batch(Part.Train).AnswerNames);
    }

    [Fact]
    public void AsAReturn_TheAnswerIsWhatTodaysPriceMadeByThen()
    {
        var answers = Priced(AheadAs.Return).Table.NumbersOf("close.ahead5");

        for (var day = 0; day < 55; day++)
        {
            Assert.Equal((Close(day + 5) / Close(day)) - 1, answers[day]!.Value, 12);
        }
    }

    [Fact]
    public void NoRowAModelLearnsFrom_ReadsItsAnswerFromTheRowsItIsMeasuredOn()
    {
        var prepared = Priced(AheadAs.Return);

        Assert.All(DaysIn(prepared, Part.Train), day => Assert.True(prepared.Parts[day + 5] is Part.Train or Part.Gap));
        Assert.All(DaysIn(prepared, Part.Validation), day => Assert.True(prepared.Parts[day + 5] is Part.Validation or Part.Gap));
        Assert.Equal(60 - 15, new[] { Part.Train, Part.Validation, Part.Test }.Sum(part => prepared.Batch(part).RowCount));
    }

    [Fact]
    public void APredictedReturn_ComesBackAsThePriceFiveDaysLater()
    {
        var prepared = Priced(AheadAs.Return);

        var back = prepared.BackToOriginal(prepared.Batch(Part.Test).Answers!, Part.Test);

        Assert.Equal(DaysIn(prepared, Part.Test).Select(day => Close(day + 5)), back.Select(price => Math.Round(price[0], 9)));
        Assert.Throws<InvalidOperationException>(() => prepared.BackToOriginal(0.01));
    }

    [Fact]
    public void AReturnPredictedForADayServedLater_ComesBackAsAPrice_FromThatDaysOwnPrice()
    {
        // The answer is made from later rows, which a served day does not have: it is never awaited from the rows
        // handed in, so a schema that refuses a column it does not name serves them all the same.
        var prepared = Priced(AheadAs.Return, Remainder.Refuse);
        var handedIn = new InMemoryRowSource(["Date", "close"], Days(3, from: 100));
        var served = prepared.Served(handedIn);

        var back = prepared.BackToOriginal([[0.1], [0.0], [-0.1]], served, handedIn);

        double[] expected = [Close(100) * 1.1, Close(101), Close(102) * 0.9];

        Assert.Equal(["close"], served.FeatureNames);
        Assert.Equal(3, back.Count);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], back[at][0], 9);
        }
    }

    [Fact]
    public void APriceAheadOfAScaledPrice_ComesBackThroughTheScaling()
    {
        var prepared = Split().Normalise("close").Ahead("close", 5, AheadAs.Value).Build().Run();

        var back = prepared.BackToOriginal(prepared.Batch(Part.Test).Answers!, Part.Test);

        Assert.Equal(DaysIn(prepared, Part.Test).Select(day => Close(day + 5)), back.Select(price => Math.Round(price[0], 9)));
    }

    [Fact]
    public void AReturnOnAPriceSomethingChangedAboveIt_IsRefused()
    {
        // The way back multiplies the price as it was read, so the return has to be made from that price.
        var refused = Assert.Throws<DeclarationException>(() => Split().Normalise("close").Ahead("close", 5, AheadAs.Return));

        var fault = Assert.Single(refused.Faults);

        Assert.Equal("target.ahead", fault.Verb);
        Assert.Contains("as it was read", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWayBackThatDoesNotLeadToThePriceLater_IsRefusedByTheRun()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .Read(new InMemoryRowSource(["Date", "close"], Days(60)), "sixty days")
                .Declare(schema => schema.Timestamp("Date").Number("close"))
                .Add(new Doubled("close"))
                .OrderBy("Date")
                .SplitByTime("Date", 0.70, 0.15, 5)
                .Drop("Date")
                .Ahead("close", 5)
                .Build()
                .Run());

        // The first day's answer is the doubled price five days on, and comes back as that; the price then was 104.
        Assert.Contains("The way back for 'close.ahead5' does not lead back", refused.Message, StringComparison.Ordinal);
        Assert.Contains("row 6, 5 rows later, was 104", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ReadingAheadAcrossASplitThatIsNotInTime_IsRefused()
    {
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create()
                .Read(new InMemoryRowSource(["Date", "close"], Days(60)), "sixty days")
                .Declare(schema => schema.Timestamp("Date").Number("close"))
                .OrderBy("Date")
                .SplitAtRandom(0.70, 0.15)
                .Ahead("close", 5));

        Assert.Contains("split.byTime", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AGapNarrowerThanHowFarTheAnswerReads_IsRefused()
    {
        var refused = Assert.Throws<DeclarationException>(() => Split(gap: 2).Ahead("close", 5));

        Assert.Contains("a gap of 2", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RowsOrderedByMoreThanTheSplitsColumn_AreRefused()
    {
        // Later rows are later in time only when the rows are ordered by the column the split divides by, alone.
        var refused = Assert.Throws<DeclarationException>(() => Split(5, Remainder.Drop, "Date", "close").Ahead("close", 5));

        Assert.Contains("'Date' alone", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOrderThatDoesNotSayWhatItOrdersBy_CannotBeReadAheadIn()
    {
        // An order from elsewhere that keeps its columns to itself cannot show that the rows after a row came after it.
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create()
                .Read(new InMemoryRowSource(["Date", "close"], Days(60)), "sixty days")
                .Declare(schema => schema.Timestamp("Date").Number("close"))
                .Add(new OrdersSomehow())
                .SplitByTime("Date", 0.70, 0.15, 5)
                .Ahead("close", 5));

        Assert.Contains("'Date' alone", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepThatReadsAhead_IsAnOutput_OrNothing()
    {
        // A feature that knows the future is a leak in mathematical dress.
        var refused = Assert.Throws<DeclarationException>(() => Split().Add(new PeeksAhead()));

        Assert.Contains("only an output", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOutputThatActs_MakesItsAnswer()
    {
        var refused = Assert.Throws<DeclarationException>(() => Split().Add(new ActsAndNames()));

        Assert.Contains("makes its answer", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAnswerAheadIsWrittenDown_AndReadBack()
    {
        var declaration = Split().Ahead("close", 5, AheadAs.Return).Declaration;
        var ahead = (AheadStep)declaration.Steps[^1];

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
        Assert.Equal(["close.ahead5"], ahead.Answers);
        Assert.Equal(5, ahead.Ahead);
        Assert.Equal(AheadAs.Return, ahead.As);
        Assert.True(((INamesTheAnswer)ahead).MakesItsAnswer);
        Assert.False(((INamesTheAnswer)new TargetStep("close")).MakesItsAnswer);
        Assert.Throws<ArgumentOutOfRangeException>(() => new AheadStep("close", 0));
        Assert.Throws<InvalidOperationException>(() => ahead.Undo(0.1, null));
        Assert.Equal(0.1, new AheadStep("close", 5).Undo(0.1, null));
    }

    /// <summary>An order from elsewhere that does not say which columns it orders by.</summary>
    private sealed class OrdersSomehow : IOrdersRows
    {
        public string Verb => "test.orders";

        public IReadOnlyList<int> RowOrder(Table table) => [.. Enumerable.Range(0, table.RowCount)];

        public void WriteTo(System.Text.Json.Utf8JsonWriter writer) =>
            throw new NotSupportedException("A test step is never written down.");
    }

    /// <summary>A feature that reads the row after its own.</summary>
    private sealed class PeeksAhead : IAddsColumns, IReadsRowsAhead
    {
        public string Verb => "test.peeks";

        public int Ahead => 1;

        public void AddTo(Table table)
        {
        }

        public void WriteTo(System.Text.Json.Utf8JsonWriter writer) =>
            throw new NotSupportedException("A test step is never written down.");
    }

    /// <summary>An output that acts on the rows without making its answer.</summary>
    private sealed class ActsAndNames : IAddsColumns, INamesTheAnswer
    {
        public string Verb => "test.acts";

        public IReadOnlyList<string> Answers => ["close"];

        public void AddTo(Table table)
        {
        }

        public void WriteTo(System.Text.Json.Utf8JsonWriter writer) =>
            throw new NotSupportedException("A test step is never written down.");
    }

    /// <summary>A step that doubles a column and claims to undo itself by doing nothing.</summary>
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
}
