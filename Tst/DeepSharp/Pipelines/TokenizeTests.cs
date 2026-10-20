// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A column of text cut into the ids an embedding takes: the vocabulary learned from the training rows alone, the place nought kept for
/// padding and the place one for a word the training rows never held, and every row written as ids of one length.
/// </summary>
public class TokenizeTests
{
    private static readonly string[] Reviews =
    [
        "good film, GOOD plot",
        "good film",
        "a plot",
        "terrible terrible terrible terrible terrible",
        "good terrible",
        string.Empty,
    ];

    // The first three rows train; the rest are for validation and test, and may hold what the training rows never did.
    private static readonly Part[] Parts = [Part.Train, Part.Train, Part.Train, Part.Validation, Part.Test, Part.Test];

    private static Table Reviewed() => new([new TextColumn("review", Reviews.Select(review => review == string.Empty ? null : review))]);

    private static float[] IdsOf(Table table, int row) => ((MediaColumn)table["review"]).ValuesAt(row)!;

    [Fact]
    public void TheVocabulary_IsWhatTheTrainingRowsHeld_MostOftenFirst_AndWhatOnlyTheOthersHeldIsNotIn()
    {
        var learned = new TokenizeStep("review", 5, 20).Fit(Reviewed(), Parts);

        // Counted over the training rows alone: good three times, film twice, plot twice, a once; 'terrible' lives on rows that do not train.
        Assert.Equal(["good", "film", "plot", "a"], learned.List("tokens"));
    }

    [Fact]
    public void TheVocabularySize_CountsPaddingAndTheUnseenPlace_AndKeepsTheMostFrequentOnly()
    {
        var learned = new TokenizeStep("review", 5, 4).Fit(Reviewed(), Parts);

        Assert.Equal(["good", "film"], learned.List("tokens"));
    }

    [Fact]
    public void EachRow_IsWrittenAsIdsOfOneLength_WordsFromTwo_UnseenAsOne_PaddingAsNought_AndNothingAsAllNought()
    {
        var table = Reviewed();
        var step = new TokenizeStep("review", 5, 20);

        step.ApplyTo(table, step.Fit(table, Parts));

        var column = Assert.IsType<MediaColumn>(table["review"]);

        Assert.True(column.Ids);
        Assert.Equal([5], column.Shape);
        Assert.Equal([2f, 3f, 2f, 4f, 0f], IdsOf(table, 0));          // good film GOOD plot — case and punctuation are not words
        Assert.Equal([5f, 4f, 0f, 0f, 0f], IdsOf(table, 2));          // a plot
        Assert.Equal([1f, 1f, 1f, 1f, 1f], IdsOf(table, 3));          // a word the training rows never held
        Assert.Equal([2f, 1f, 0f, 0f, 0f], IdsOf(table, 4));
        Assert.Equal([0f, 0f, 0f, 0f, 0f], IdsOf(table, 5));          // an empty cell is no words
    }

    [Fact]
    public void ARowLongerThanTheLength_IsCutAtIt()
    {
        var table = new TextColumn("review", ["good good good good good good"]);
        var cut = new Table([table]);
        var step = new TokenizeStep("review", 3, 5);

        step.ApplyTo(cut, step.Fit(cut, [Part.Train]));

        Assert.Equal([2f, 2f, 2f], IdsOf(cut, 0));
    }

    [Fact]
    public void ATrainingSetWithoutAWord_IsRefused()
    {
        var empty = new Table([new TextColumn("review", new string?[] { null, "..." , "!" })]);

        Assert.Contains("holds a word", Assert.Throws<InvalidOperationException>(() => new TokenizeStep("review", 5, 20).Fit(empty, [Part.Train, Part.Train, Part.Train])).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheColumn_StaysWhereItWas_AMediaColumnOfIds_ForTheStepsAfter()
    {
        var state = ColumnState.None.With("review", ColumnKind.Text).With("good", ColumnKind.Integer);
        var after = new TokenizeStep("review", 5, 20).After(state);

        Assert.Equal(ColumnKind.Media, after.Columns.Single(column => column.Name == "review").Kind);
        Assert.Equal(ColumnKind.Integer, after.Columns.Single(column => column.Name == "good").Kind);
    }

    [Fact]
    public void TheStep_IsWrittenAndReadBackAsItIs_AndRefusesWhatMakesNoVocabulary()
    {
        var step = new TokenizeStep("review", 32, 1000);

        Assert.Equal("tokenize", step.Verb);
        Assert.Equal(step, TokenizeStep.ReadFrom(JsonDocument.Parse(Encoding.UTF8.GetString(step.Canonical())).RootElement));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenizeStep("review", 0, 1000));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenizeStep("review", 32, 2));
        Assert.Throws<ArgumentException>(() => new TokenizeStep(" ", 32, 1000));
    }

    [Fact]
    public void InAPipeline_TheTextIsHandedOverAsIds_ExemptFromTheRangeNumbersLandIn_AndAServedRowIsCutByTheSameVocabulary()
    {
        var rows = new InMemoryRowSource(
            ["review", "good"],
            [.. Enumerable.Range(0, 60).Select(at => new string?[] { at % 2 == 0 ? $"good film number{at}" : $"bad film number{at}", at % 2 == 0 ? "1" : "0" })]);
        var prepared = Pdd.Create()
            .Read(rows, "sixty reviews")
            .Declare(schema => schema.Text("review").Integer("good"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Tokenize("review", length: 4, vocabulary: 8)
            .Target("good")
            .Build()
            .Run();
        var batch = prepared.Batch(Part.Train, Needs.OneScale);

        Assert.Equal(4, batch.Width);
        Assert.Equal("review.0", batch.FeatureNames[0]);
        Assert.Equal([4], batch.ExampleShape);
        Assert.Contains(batch.Features, row => row.Any(value => value > 1));
        Assert.All(batch.Features, row => Assert.All(row, value => Assert.InRange(value, 0, 7)));

        var served = prepared.Served(new InMemoryRowSource(["review", "good"], [["good film that was never seen", "1"]]), Needs.OneScale);

        Assert.Equal([4], served.ExampleShape);
        // film is in every training row (place 2), good and bad in half of them (bad before good, by letters); the rest was never seen.
        Assert.Equal([4, 2, 1, 1], served.Features[0]);
    }
}
