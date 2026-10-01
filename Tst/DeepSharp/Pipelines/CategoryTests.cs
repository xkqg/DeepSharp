// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Which columns stand for a group rather than for themselves is a fact about the data, so it is said where
/// the data is declared and not again further down. Everything after reads it from there: encoding takes
/// them by name, and a category that never became numbers is refused at the handover rather than dropped.
/// </summary>
public class CategoryTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived", "pclass")
                .Number("fare")
                .Category("sex", "embarked"));

    // Ten rows in time, eight of them training rows: the categories a, b and one more, and a gap, then two later rows.
    private static FittingBuilder Grouped(string held, string later) =>
        Pdd.Create()
            .Read(
                new InMemoryRowSource(
                    ["t", "g"],
                    [["1", "a"], ["2", held], ["3", "b"], ["4", null], ["5", "a"], ["6", "b"], ["7", held], ["8", "a"], ["9", later], ["10", "a"]]),
                "ten rows")
            .Declare(schema => schema.Integer("t").Category("g"))
            .SplitByTime("t", 0.80);

    [Fact]
    public void ACategoryCalledOther_IsRefusedByAnEncoderKeepingAPlaceOfThatNameForWhatTheTrainingRowsNeverHeld()
    {
        // The place kept for a category the training rows never held is written as g_other, and so would be the trained
        // category 'other': one would take the other's column, and its rows would lose their category without a word.
        Func<PreparedData>[] fittings =
        [
            () => Grouped("other", "zzz").EncodeCategories().Build().Run(),
            () => Grouped("other", "zzz").Encode("g").Build().Run(),
            () => Grouped("other", "zzz").EncodeCategories().Build().RunFor(Needs.Categories),
        ];

        foreach (var refused in fittings.Select(fitting => Assert.Throws<InvalidOperationException>(fitting)))
        {
            Assert.Contains("'g' holds the category 'other' on its training rows", refused.Message, StringComparison.Ordinal);
            Assert.Contains("'g_other'", refused.Message, StringComparison.Ordinal);
            Assert.Contains("unseen: refuse", refused.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ACategoryCalledOther_IsWrittenDown_WhereNoPlaceIsKeptUnderItsName()
    {
        // Refusing what the training rows never held keeps no place, and places written as numbers name none.
        var refusing = Grouped("other", "a").EncodeCategories(unseen: Unseen.Refuse).Build().Run();
        var placed = Grouped("other", "zzz").EncodeCategories(As.Ordinal).Build().Run();

        Assert.Equal(["t", "g_a", "g_b", "g_other", "g_was_missing"], refusing.Table.Columns.Select(column => column.Name));
        Assert.Equal([0, 1, 0, 0, 0, 0, 1, 0, 0, 0], Values(refusing.Table, "g_other"));
        Assert.Equal([0, 2, 1, 0, 0, 1, 2, 0, 3, 0], Values(placed.Table, "g"));
    }

    [Fact]
    public void ACategoryCalledWasMissing_IsRefusedByAnEncoderWritingOneColumnForEachCategory()
    {
        // Its column would be g_was_missing, the column that marks where the cell was empty.
        foreach (var unseen in new[] { Unseen.Reserve, Unseen.Refuse })
        {
            var refused = Assert.Throws<InvalidOperationException>(() => Grouped("was_missing", "a").EncodeCategories(unseen: unseen).Build().Run());

            Assert.Contains("'g' holds the category 'was_missing' on its training rows", refused.Message, StringComparison.Ordinal);
            Assert.Contains("'g_was_missing'", refused.Message, StringComparison.Ordinal);
        }

        Assert.Equal(["t", "g", "g_was_missing"], Grouped("was_missing", "a").EncodeCategories(As.Ordinal).Build().Run().Table.Columns.Select(column => column.Name));
    }

    [Fact]
    public void AFileWhoseEncoderLearnedACategoryCalledOther_IsReplayedAsItWasWritten()
    {
        // A fit before this refusal could learn 'other', and a file it wrote replays as it always did: nothing is fitted in
        // a replay, so nothing is refused there, and a served row lands where that file's network was trained to take it.
        var written = Grouped("x", "zzz").EncodeCategories().Build().Run().ToJson().Replace("\"x\"", "\"other\"", StringComparison.Ordinal);
        var loaded = PreparedData.FromJson(written, StepCatalog.BuiltIn());

        var replayed = loaded.Replay(new InMemoryRowSource(["t", "g"], [["11", "other"], ["12", "zzz"], ["13", "a"]]));

        Assert.Equal(["a", "b", "other"], loaded.Fitted[loaded.Declaration.Steps.Count - 1].List("g"));
        Assert.Equal(["t", "g_a", "g_b", "g_other", "g_was_missing"], replayed.Columns.Select(column => column.Name));
        Assert.Equal([0, 1, 0], Values(replayed, "g_other"));
    }

    private static double[] Values(Table table, string column) =>
        [.. Enumerable.Range(0, table.RowCount).Select(row => ((Column<double>)table[column])[row]!.Value)];

    [Fact]
    public void AColumnSaidToBeACategory_IsOneFromTheMomentItIsRead()
    {
        var table = Passengers().Build().Prepare();

        Assert.Equal(ColumnKind.Category, table["sex"].Kind);
        Assert.Equal(ColumnKind.Category, table["embarked"].Kind);
        Assert.Equal(ColumnKind.Number, table["fare"].Kind);
        Assert.Equal("male", ((TextColumn)table["sex"])[0]);
    }

    [Fact]
    public void TheSchemaKnowsWhichOnesTheyWere()
    {
        var declared = (DeclareStep)Passengers().Declaration.Steps[1];

        Assert.Equal(["sex", "embarked"], declared.Categories);
    }

    [Fact]
    public void EncodingTakesThemByNameRatherThanAskingAgain()
    {
        var prepared = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Build()
            .Run();

        Assert.False(prepared.Table.Has("sex"));
        Assert.False(prepared.Table.Has("embarked"));
        Assert.True(prepared.Table.Has("sex_female"));
        Assert.True(prepared.Table.Has("embarked_S"));

        // One step, written as itself: which columns it encodes is decided where it stands, from what the
        // schema declared, so a file can say it and a column marked a category later is encoded too.
        Assert.Equal("encode.categories", prepared.Declaration.Steps[^1].Verb);
    }

    [Fact]
    public void AndTheResultIsHandedOverAsNumbers()
    {
        var batch = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories(As.Ordinal)
            .Target("survived")
            .Build()
            .Run()
            .Batch(Part.Train);

        Assert.Contains("sex", batch.FeatureNames);
        Assert.Equal(batch.Width, batch.Features[0].Length);
    }

    [Fact]
    public void ACategoryNobodyEncoded_IsRefusedAtTheHandoverRatherThanDropped()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Passengers()
                .SplitStratified("survived", 0.70, 0.15)
                .Build()
                .Run()
                .Batch(Part.Train));

        Assert.Contains("sex", refused.Message);
        Assert.Contains("Encode it", refused.Message);
    }

    [Fact]
    public void ASchemaWithoutCategories_SaysSoRatherThanQuietlyDoingNothing()
    {
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Integer("survived"))
                .SplitAtRandom(0.70, 0.15)
                .EncodeCategories());

        Assert.Contains("no categories", refused.Message);
    }

    [Fact]
    public void ACategorySurvivesTheFile()
    {
        var declaration = Passengers().Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal(["sex", "embarked"], ((DeclareStep)returned.Steps[1]).Categories);
    }

    [Fact]
    public void AColumnOfWordsIsTextOrACategory_AndNothingElse()
    {
        Assert.Throws<ArgumentException>(() => new TextColumn("a", ColumnKind.Number, ["x"]));
        Assert.Equal(ColumnKind.Text, new TextColumn("a", ["x"]).Kind);
    }
}
