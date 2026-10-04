// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A fit writes down three kinds of thing, and only two of them are replayed.
/// </summary>
/// <remarks>
/// It learns a value — a mean, a bound, the categories it found. It decides something from what it saw — whether a column
/// with that many gaps is filled at all. And it counts what it saw: how many gaps there were, how many values were no
/// number, how many lay outside the bounds. The first two are read when the pipeline is replayed; the third is evidence,
/// written down so a person can see what the fit was looking at, and no replay may depend on it. Saying them apart in one
/// bag is the whole of it: the file holds one object of names and values, the same bytes in the same order, so every
/// pipeline already published keeps loading.
/// </remarks>
public class FittedVocabularyTests
{
    // Every fit there is, and the whole of what it writes down: what it counted, what it decided, and what it learned.
    public static TheoryData<string, string[], string[], string[]> EveryFit() => new()
    {
        { "normalise", [], [], ["centre", "spread"] },
        { "normalise.power", [], [], ["centre", "lambda", "spread"] },
        { "encode", [], [], ["categories"] },
        { "encode.categories", [], [], ["c"] },
        { "fill.missing", ["gaps"], [], ["value"] },
        { "fill.missing.refused", ["gaps", "share"], ["filled"], [] },
        { "fill.nan", ["notNumbers"], [], [] },
        { "fill.nan.mean", ["notNumbers"], [], ["value"] },
        { "clip", ["outside"], [], ["lower", "upper"] },
    };

    [Theory]
    [MemberData(nameof(EveryFit))]
    public void WhatEachFitWritesDown_IsSaidUnderTheWordItIs(string fit, string[] counted, string[] decided, string[] learned)
    {
        var (step, table) = Of(fit);
        var values = step.Fit(table, Training(table));

        Assert.Equal(counted, values.Counts.Order(StringComparer.Ordinal));
        Assert.Equal(decided, values.Decisions.Order(StringComparer.Ordinal));
        Assert.Equal(
            learned,
            values.Numbers.Keys.Concat(values.Lists.Keys).Concat(values.Curves.Keys).Concat(values.Texts.Keys)
                .Except(values.Counts).Except(values.Decisions).Order(StringComparer.Ordinal));
    }

    [Theory]
    [MemberData(nameof(EveryFit))]
    public void WhatAFitOnlyCounted_IsReadByNoReplay(string fit, string[] counted, string[] decided, string[] learned)
    {
        // Every count taken out of the bag, and the step applied again: a replay that read one would come out differently,
        // and the file it was read from need not hold it at all.
        _ = (decided, learned);
        var (step, table) = Of(fit);
        var values = step.Fit(table, Training(table));

        Assert.Equal(counted, values.Counts.Order(StringComparer.Ordinal));
        Assert.Equal(Shown(step, table, values), Shown(step, table, Apart(values, values.Counts)));
    }

    [Theory]
    [MemberData(nameof(EveryFit))]
    public void WhatAFitLearnedOrDecided_IsReadByTheReplay(string fit, string[] counted, string[] decided, string[] learned)
    {
        // The mirror, so the three words are not a label anybody can hang where they like: take away what was learned or
        // decided and the replay cannot go on, or goes on differently.
        _ = counted;
        var (step, table) = Of(fit);
        var values = step.Fit(table, Training(table));

        foreach (var name in decided.Concat(learned))
        {
            Assert.True(
                Shown(step, table, Apart(values, [name])) != Shown(step, table, values),
                $"'{name}' was taken out of what '{fit}' wrote down, and the replay came out the same, so nothing reads it.");
        }
    }

    [Fact]
    public void EveryStepThatLearns_HasItsWordsPinnedHere()
    {
        // The theories above are only worth anything if they cover every step that writes into the fitted half, so the
        // set is taken from the catalog rather than typed: a verb that learns and is not exercised here says so the day
        // it is registered, and a step that lied about what it learned would be measured by nothing.
        var learns = Shipped.Catalog().Descriptions
            .Select(description => Shipped.Catalog().ReadStep(description.Template))
            .OfType<IFittedStep>()
            .Select(step => ((IPipelineStep)step).Verb)
            .ToHashSet(StringComparer.Ordinal);

        var exercised = new[]
            {
                "normalise", "normalise.power", "encode", "encode.categories", "fill.missing", "fill.missing.refused",
                "fill.nan", "fill.nan.mean", "clip",
            }
            .Select(fit => ((IPipelineStep)Of(fit).Step).Verb)
            .ToHashSet(StringComparer.Ordinal);

        Assert.Equal(learns.Order(StringComparer.Ordinal), exercised.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void NoStepBothLearnsAndOnlyAddsColumns()
    {
        // The two capabilities answer the same question differently: one is fitted on the training rows and replayed, the
        // other does the same thing wherever it stands. A step that declared both would be placed by the rule as a
        // learner and acted on as neither.
        Assert.All(
            Shipped.Catalog().Descriptions.Select(description => Shipped.Catalog().ReadStep(description.Template)),
            step => Assert.False(step is IFittedStep and IAddsColumns, $"'{step.Verb}' says it both learns and only adds columns."));
    }

    [Fact]
    public void TheThreeWords_WriteTheOneObjectTheFileAlwaysHeld()
    {
        // A number is a number in the file whichever word wrote it down: the same names in the same order, so the digest a
        // trained model is bound to does not move.
        var said = new FittedStepValues();
        said.Learned("value", 29.5);
        said.Saw("gaps", 133);
        said.Decided("filled", 1);

        var plainly = new FittedStepValues();
        plainly.Learned("value", 29.5);
        plainly.Learned("gaps", 133);
        plainly.Learned("filled", 1);

        Assert.Equal(Written(plainly), Written(said));
        Assert.Equal(["gaps"], said.Counts);
        Assert.Equal(["filled"], said.Decisions);
        Assert.Equal(133, said.Number("gaps"));
    }

    [Fact]
    public void AFitReadBackFromAFile_CountsNothing()
    {
        // The file marks no name, so a pipeline loaded from one cannot say which of its numbers were counts. The words are
        // the fit's own, and what a replay reads is the same either way.
        var written = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Build()
            .Run();

        Assert.Equal(["gaps"], written.Fitted[3].Counts);

        var read = PreparedData.FromJson(written.ToJson(), StepCatalog.BuiltIn());

        Assert.Empty(read.Fitted[3].Counts);
        Assert.Empty(read.Fitted[3].Decisions);
        Assert.Equal(written.Fitted[3].Number("gaps"), read.Fitted[3].Number("gaps"));
    }

    [Fact]
    public void ANameSaidTwoWays_IsTheWayItWasSaidLast()
    {
        // One name holds one number, and the word it was written down under is the word of the writing that stands.
        var values = new FittedStepValues();
        values.Saw("gaps", 3);
        values.Learned("gaps", 4);

        Assert.Equal(4, values.Number("gaps"));
        Assert.Empty(values.Counts);

        values.Decided("gaps", 5);

        Assert.Equal(["gaps"], values.Decisions);
        Assert.Empty(values.Counts);
        values.Learned("categories", ["a", "b"]);

        Assert.Throws<ArgumentException>(() => values.Saw("categories", 1));
    }

    // Every row of the table is a training row: a part for each, which is how a fit is told which rows it may learn from.
    private static Part[] Training(Table table) => [.. Enumerable.Repeat(Part.Train, table.RowCount)];

    // One step of each shape there is, and a table it has something to learn from.
    private static (IFittedStep Step, Table Table) Of(string fit) => fit switch
    {
        "normalise" => (new NormaliseStep("a"), Numbers(1, 2, 3, 4, 10)),
        "normalise.power" => (new NormaliseStep("a", Scale.Power), Numbers(1, 2, 3, 4, 10)),
        "encode" => (new EncodeStep("c"), Words("x", "y", "x")),
        "encode.categories" => (new EncodeCategoriesStep(), Words("x", "y", "x")),
        "fill.missing" => (FillMissingStep.Of("a", With.Median), Numbers(1, 2, null, 4, 10)),
        "fill.missing.refused" => (FillMissingStep.Of("a", With.Median, 0.10), Numbers(1, 2, null, 4, 10)),
        "fill.nan" => (new FillNaNStep("a"), Numbers(1, 2, 3, 4, 10)),
        "fill.nan.mean" => (new FillNaNStep("a", With.Mean), Numbers(1, 2, double.NaN, 4, 10)),
        _ => (new ClipOutliersStep("a"), Numbers(1, 2, 3, 4, 100)),
    };

    private static Table Numbers(params double?[] values) =>
        new([new Column<double>("a", ColumnKind.Number, values)]);

    private static Table Words(params string?[] values) =>
        new([new TextColumn("c", ColumnKind.Category, values)]);

    // What the step leaves of the table, as text: the whole of what a replay of it does.
    private static string Shown(IFittedStep step, Table table, FittedStepValues fitted)
    {
        var applied = new Table(table.Columns.Select(column => column.Rows([.. Enumerable.Range(0, table.RowCount)])));

        try
        {
            step.ApplyTo(applied, fitted);
        }
        catch (InvalidOperationException refused)
        {
            return refused.Message;
        }

        return string.Join(
            " | ",
            applied.Columns.Select(column => $"{column.Name}: {string.Join(",", Enumerable.Range(0, applied.RowCount).Select(column.TextAt))}"));
    }

    // The same bag without the names given: what a file that never held them would be read back as.
    private static FittedStepValues Apart(FittedStepValues values, IEnumerable<string> names)
    {
        var apart = new FittedStepValues();
        var gone = names.ToHashSet(StringComparer.Ordinal);

        foreach (var (name, value) in values.Numbers.Where(each => !gone.Contains(each.Key)))
        {
            apart.Learned(name, value);
        }

        foreach (var (name, list) in values.Lists.Where(each => !gone.Contains(each.Key)))
        {
            apart.Learned(name, list);
        }

        foreach (var (name, curve) in values.Curves.Where(each => !gone.Contains(each.Key)))
        {
            apart.Learned(name, curve);
        }

        foreach (var (name, text) in values.Texts.Where(each => !gone.Contains(each.Key)))
        {
            apart.Learned(name, text);
        }

        return apart;
    }

    private static string Written(FittedStepValues values)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            values.WriteTo(writer);
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
