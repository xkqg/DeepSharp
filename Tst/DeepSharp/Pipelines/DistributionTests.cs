// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// An answer that is how a whole is divided: a flock weighed in bands of fifty grams is seventy numbers, the share
/// of its birds in each band. The shares are one answer, handed over in their order and held to summing to one; with
/// the column saying how many birds there were, they come back as the birds in each band.
/// </summary>
public class DistributionTests
{
    // How many birds of flock f fell in band b: a pattern, so every band of every flock holds some.
    private static int Birds(int flock, int band) => 1 + (((flock * 7) + (band * 3)) % 5);

    private static string[] Bands(int bands) => [.. Enumerable.Range(0, bands).Select(band => $"w{500 + (50 * band)}")];

    // Six flocks: their age, how many birds each had, and how many of them fell in each band.
    private static InMemoryRowSource Flocks(int bands, int birdsOffBy = 0) =>
        new(
            ["age", "chicks", .. Bands(bands)],
            [
                .. Enumerable.Range(0, 6).Select(flock => (IReadOnlyList<string?>)
                [
                    (30 + flock).ToString(CultureInfo.InvariantCulture),
                    (Enumerable.Range(0, bands).Sum(band => Birds(flock, band)) + (flock == 2 ? birdsOffBy : 0)).ToString(CultureInfo.InvariantCulture),
                    .. Enumerable.Range(0, bands).Select(band => Birds(flock, band).ToString(CultureInfo.InvariantCulture)),
                ]),
            ]);

    private static PreparedData Weighed(int bands, bool scaled = true, bool divided = true, int birdsOffBy = 0)
    {
        var names = Bands(bands);
        var fitting = Pdd.Create()
            .Read(Flocks(bands, birdsOffBy), "six flocks")
            .Declare(schema => schema.Number("age", "chicks").Number(names))
            .SplitAtRandom(0.50, seed: 3);

        if (divided)
        {
            fitting.NormaliseRow(Norm.L1, names);
        }

        return fitting.Distribution(names, scaled ? "chicks" : null).Build().Run();
    }

    [Fact]
    public void SeventyBandsOfFiftyGrams_AreOneAnswer_HandedOverAsSharesInTheirOrder()
    {
        var batch = Weighed(70).Batch(Part.Train);

        Assert.Equal(70, batch.AnswerNames!.Count);
        Assert.Equal("w500", batch.AnswerNames[0]);
        Assert.Equal("w3950", batch.AnswerNames[^1]);
        Assert.Equal(["age", "chicks"], batch.FeatureNames);
        Assert.All(batch.Answers!, shares => Assert.Equal(1, shares.Sum(), 9));
        Assert.Null(batch.Labels);
    }

    [Fact]
    public void Shares_ComeBackAsTheBirdsInEachBand_ByTheFlockOfTheirOwnRow()
    {
        var prepared = Weighed(3);
        var batch = prepared.Batch(Part.Test);

        var back = prepared.BackToOriginal(batch.Answers!, Part.Test);
        var flocks = prepared.Table.Identities.Where((_, row) => prepared.Parts[row] == Part.Test).Select(identity => identity.ReadAt).ToArray();

        Assert.Equal(flocks.Length, back.Count);

        for (var at = 0; at < flocks.Length; at++)
        {
            for (var band = 0; band < 3; band++)
            {
                Assert.Equal(Birds(flocks[at], band), back[at][band], 9);
            }
        }
    }

    [Fact]
    public void SharesPredictedForAFlockServedLater_ComeBackAsBirds_ByTheFlockHandedIn()
    {
        var prepared = Weighed(3);
        var handedIn = new InMemoryRowSource(["age", "chicks"], [["35", "1000"]]);
        var served = prepared.Served(handedIn);

        var back = prepared.BackToOriginal([[0.2, 0.5, 0.3]], served, handedIn);

        Assert.Equal(["age", "chicks"], served.FeatureNames);
        Assert.Equal([200.0, 500.0, 300.0], back[0].Select(birds => Math.Round(birds, 9)));
    }

    [Fact]
    public void BandsThatDoNotAddUpToTheFlock_AreRefusedByTheRun()
    {
        // Coming back as birds says the bands add up to the flock; the run tries it on every training row, and a
        // flock with one bird more than its bands is caught there rather than in a report.
        var refused = Assert.Throws<InvalidOperationException>(() => Weighed(3, birdsOffBy: 1));

        Assert.Contains("does not lead back", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SharesThatDoNotSumToOne_AreRefusedAtTheHandover_SayingWhatDividesThem()
    {
        var prepared = Weighed(3, scaled: false, divided: false);

        var refused = Assert.Throws<InvalidOperationException>(
            () => new[] { Part.Train, Part.Test }.Select(part => prepared.Batch(part)).ToArray());

        Assert.Contains("'target.distribution'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("normalise.row", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AShareBelowNought_IsRefused_ThoughTheSharesSumToOne()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["age", "w500", "w550"], [["30", "1.5", "-0.5"], ["31", "0.5", "0.5"]]), "two flocks")
            .Declare(schema => schema.Number("age", "w500", "w550"))
            .SplitAtRandom(0.50, seed: 3)
            .Distribution(["w500", "w550"])
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(
            () => new[] { Part.Train, Part.Test }.Select(part => prepared.Batch(part)).ToArray());

        Assert.Contains("below nought", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutWhatTheSharesAreSharesOf_TheyComeBackAsShares()
    {
        var prepared = Weighed(3, scaled: false);
        var batch = prepared.Batch(Part.Test);

        var back = prepared.BackToOriginal(batch.Answers!, Part.Test);

        Assert.Equal(batch.Answers!.SelectMany(shares => shares), back.SelectMany(shares => shares));
        Assert.Throws<InvalidOperationException>(() => prepared.BackToOriginal(0.5));
    }

    [Fact]
    public void BirdsFromAShareNeedTheirRow()
    {
        var step = new DistributionStep(["w500", "w550"], "chicks");

        var refused = Assert.Throws<InvalidOperationException>(() => step.Undo(0.5, null));

        Assert.Contains("only with the row", refused.Message, StringComparison.Ordinal);
        Assert.Equal(0.5, new DistributionStep(["w500", "w550"]).Undo(0.5, null));
    }

    [Fact]
    public void ADistributionIsDividedAmongAtLeastTwoColumns()
    {
        Assert.Throws<ArgumentException>(() => new DistributionStep(["w500"]));
        Assert.Throws<ArgumentNullException>(() => new DistributionStep(null!));
    }

    [Fact]
    public void ANewBlockOfADistribution_StartsWithoutWhatItsSharesAreSharesOf_AndAFileMayLeaveItOut()
    {
        // What the shares are shares of is a column the step can do without: a new block starts with none, which a file
        // says by leaving the key out, and the shares then come back as shares.
        var catalog = StepCatalog.BuiltIn();
        var template = catalog.Describe("target.distribution").Template;
        var read = (DistributionStep)catalog.ReadStep("""{"step": "target.distribution", "columns": ["w500", "w550"]}""");

        Assert.DoesNotContain("scaleBy", template, StringComparison.Ordinal);
        Assert.Null(read.ScaleBy);
        Assert.Empty(catalog.Describe("target.distribution").Parameters.Single(parameter => parameter.Key == "scaleBy").RequiredKeys);
    }

    [Fact]
    public void WhatTheSharesAreSharesOf_IsNoneOfTheShares_InCodeAndInAFile()
    {
        var refused = Assert.Throws<ArgumentException>(() => new DistributionStep(["w500", "w550"], scaleBy: "w550"));
        var inAFile = Assert.Throws<PipelineFileException>(() =>
            StepCatalog.BuiltIn().ReadStep("""{"step": "target.distribution", "columns": ["w500", "w550"], "scaleBy": "w500"}"""));

        Assert.Contains("'w550' is one of the shares", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'w500' is one of the shares", inAFile.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADistributionIsWrittenDown_WithWhatItsSharesAreSharesOf_OnlyWhenThereIsSuch()
    {
        var shares = new DistributionStep(["w500", "w550"]);
        var birds = new DistributionStep(["w500", "w550"], "chicks");
        var catalog = StepCatalog.BuiltIn();

        var declaration = new PipelineDeclaration(
        [
            new ReadCsvStep("flocks.csv"),
            new DeclareStep([.. new[] { "chicks", "w500", "w550" }.Select(name => new ColumnDeclaration(name, ColumnKind.Number, Optional: false))]),
            new SplitAtRandomStep(new SplitShares(0.50, 0, 0.50), 3),
            birds,
        ]);

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), catalog));
        Assert.Contains("\"scaleBy\":\"chicks\"", declaration.ToJson().Replace(" ", string.Empty, StringComparison.Ordinal), StringComparison.Ordinal);
        Assert.DoesNotContain("scaleBy", new PipelineDeclaration([.. declaration.Steps.Take(3), shares]).ToJson(), StringComparison.Ordinal);
        Assert.NotEqual(shares, birds);
        Assert.Equal(shares, new DistributionStep(["w500", "w550"]));
        Assert.Equal(shares.GetHashCode(), new DistributionStep(["w500", "w550"]).GetHashCode());
        Assert.Equal(["w500", "w550"], shares.Answers);
        Assert.Null(shares.ScaleBy);
        Assert.Equal("w500", shares.Produces);
    }

    [Fact]
    public void ADistributionWhoseBandsAreInAnOrder_SaysSo_AndIsWrittenSoOnlyThen()
    {
        // Bands of weight are in an order that means something; the shares of a flock among its farms are not. Left out, a
        // distribution is not ordered, which is what every file written before the key existed meant.
        var ordered = new DistributionStep(["w500", "w550"], "chicks", ordered: true);
        var unordered = new DistributionStep(["w500", "w550"], "chicks");
        var catalog = StepCatalog.BuiltIn();

        Assert.True(ordered.Ordered);
        Assert.True(((INamesTheAnswer)ordered).IsOrdered);
        Assert.False(((INamesTheAnswer)unordered).IsOrdered);
        Assert.False(((INamesTheAnswer)new TargetStep("price")).IsOrdered);
        Assert.False(((INamesTheAnswer)new LabelsStep(["cat", "dog"])).IsOrdered);
        Assert.NotEqual(ordered, unordered);
        Assert.NotEqual(ordered.GetHashCode(), unordered.GetHashCode());
        Assert.Equal(ordered, catalog.ReadStep("""{"step": "target.distribution", "columns": ["w500", "w550"], "scaleBy": "chicks", "ordered": true}"""));
        Assert.Equal(unordered, catalog.ReadStep("""{"step": "target.distribution", "columns": ["w500", "w550"], "scaleBy": "chicks", "ordered": false}"""));
        Assert.Contains("\"ordered\":true", Written(ordered), StringComparison.Ordinal);
        Assert.DoesNotContain("ordered", Written(unordered), StringComparison.Ordinal);
        Assert.DoesNotContain("ordered", catalog.Describe("target.distribution").Template, StringComparison.Ordinal);
        Assert.Empty(catalog.Describe("target.distribution").Parameters.Single(parameter => parameter.Key == "ordered").RequiredKeys);
    }

    [Fact]
    public void TheChainSaysADistributionIsOrdered_AndWhatIsLeftOfItsWhole_InOneDoor()
    {
        var names = Bands(3);
        var output = Pdd.Create()
            .Read(Flocks(3), "six flocks")
            .Declare(schema => schema.Number("age", "chicks").Number(names))
            .SplitAtRandom(0.50, seed: 3)
            .Distribution(names, "chicks", ordered: true, remainder: "other")
            .Build()
            .Declaration.Output;

        Assert.Equal(new DistributionStep(names, "chicks", ordered: true, remainder: "other"), output);
    }

    // Six flocks planned at a number of chicks, and weighed in bands as they arrived: some fewer than planned.
    private static InMemoryRowSource Arrived(int bands, int moreThanPlannedAt = -1) =>
        new(
            ["age", "planned", .. Bands(bands)],
            [
                .. Enumerable.Range(0, 6).Select(flock => (IReadOnlyList<string?>)
                [
                    (30 + flock).ToString(CultureInfo.InvariantCulture),
                    (Enumerable.Range(0, bands).Sum(band => Birds(flock, band)) + (flock == moreThanPlannedAt ? -1 : flock % 3)).ToString(CultureInfo.InvariantCulture),
                    .. Enumerable.Range(0, bands).Select(band => Birds(flock, band).ToString(CultureInfo.InvariantCulture)),
                ]),
            ]);

    private static PreparedData Planned(int bands, int moreThanPlannedAt = -1) =>
        Pdd.Create()
            .Read(Arrived(bands, moreThanPlannedAt), "six flocks")
            .Declare(schema => schema.Number("age", "planned").Number(Bands(bands)))
            .SplitAtRandom(0.50, seed: 3)
            .Distribution(Bands(bands), "planned", ordered: true, remainder: "other")
            .Build()
            .Run();

    [Fact]
    public void AFlockThatArrivedShortOfPlanned_IsOneMoreAnswer_WhatIsLeft_AndTheSharesAreOfThePlannedWhole()
    {
        // The bands hold the birds that arrived; what is left of the planned flock is one more answer, after the bands, so
        // the bands and what is left are shares of the planned flock, and they come back as birds that add up to it.
        var prepared = Planned(3);
        var batch = prepared.Batch(Part.Train);
        var flocks = prepared.Table.Identities.Where((_, row) => prepared.Parts[row] == Part.Train).Select(identity => identity.ReadAt).ToArray();

        Assert.Equal([.. Bands(3), "other"], batch.AnswerNames);
        Assert.Equal(["age", "planned"], batch.FeatureNames);
        Assert.All(batch.Answers!, shares => Assert.Equal(1, shares.Sum(), 12));

        var back = prepared.BackToOriginal(batch.Answers!, Part.Train);

        for (var at = 0; at < flocks.Length; at++)
        {
            var planned = Enumerable.Range(0, 3).Sum(band => Birds(flocks[at], band)) + (flocks[at] % 3);

            Assert.Equal(planned, back[at].Sum(), 9);
            Assert.Equal(flocks[at] % 3, back[at][3], 9);

            for (var band = 0; band < 3; band++)
            {
                Assert.Equal(Birds(flocks[at], band), back[at][band], 9);
            }
        }
    }

    [Fact]
    public void AFlockThatArrivedLargerThanPlanned_IsRefusedInWords_NamingItsRow()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Planned(3, moreThanPlannedAt: 4));

        Assert.Contains("Row 5", refused.Message, StringComparison.Ordinal);
        Assert.Contains("more than", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'planned'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFlockPlannedAtNone_IsRefusedInWords_NamingItsRow()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["age", "planned", "w500", "w550"], [["30", "0", "0", "0"], ["31", "4", "1", "2"]]), "two flocks")
            .Declare(schema => schema.Number("age", "planned", "w500", "w550"))
            .SplitAtRandom(0.50, seed: 3)
            .Distribution(["w500", "w550"], "planned", ordered: true, remainder: "other")
            .Build();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Run());

        Assert.Contains("Row 1", refused.Message, StringComparison.Ordinal);
        Assert.Contains("nought", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFlockWithAGapInItsBands_KeepsTheGap_ForTheHandoverToRefuse()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["age", "planned", "w500", "w550"], [["30", "4", "", "2"], ["31", "4", "1", "2"]]), "two flocks")
            .Declare(schema => schema.Number("age", "planned", "w550").Optional("w500", ColumnKind.Number))
            .SplitAtRandom(0.50, seed: 3)
            .Distribution(["w500", "w550"], "planned", ordered: true, remainder: "other")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(
            () => new[] { Part.Train, Part.Test }.Select(part => prepared.Batch(part)).ToArray());

        Assert.Contains("Row 1", refused.Message, StringComparison.Ordinal);
        Assert.Contains("gap", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatIsLeftOfTheWhole_IsMadeFromTheCountsAsTheyWereRead_SoAStepAboveThatChangesThemIsRefused()
    {
        // A scale learned on the planned flock, above the output, would leave it a number of no birds to divide by; it is
        // refused where the pipeline is declared, naming the step.
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create()
            .Read(Arrived(3), "six flocks")
            .Declare(schema => schema.Number("age", "planned").Number(Bands(3)))
            .SplitAtRandom(0.50, seed: 3)
            .Normalise("planned", Scale.MinMax)
            .Distribution(Bands(3), "planned", ordered: true, remainder: "other")
            .Build());

        Assert.Contains("'planned'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("step 4", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'other'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BandsDividedByTheirSumAboveWhatIsLeftOfTheWhole_DoNotLeadBack_AndTheRunSaysSo()
    {
        // Shares of what arrived are no counts: over the planned flock they would be shares of shares.
        var refused = Assert.Throws<InvalidOperationException>(() => Pdd.Create()
            .Read(Arrived(3), "six flocks")
            .Declare(schema => schema.Number("age", "planned").Number(Bands(3)))
            .SplitAtRandom(0.50, seed: 3)
            .NormaliseRow(Norm.L1, Bands(3))
            .Distribution(Bands(3), "planned", ordered: true, remainder: "other")
            .Build()
            .Run());

        Assert.Contains("does not lead back", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatIsLeftOfTheWhole_IsTheColumnTheOutputSaysItMakes_AndTheBandsBecomeNumbers()
    {
        // The description and the act are two things that could drift: the bands, counted in whole birds, become shares, and
        // what is left is one column more.
        var table = SchemaBinding.Bind(
            new DeclareStep([
                new ColumnDeclaration("planned", ColumnKind.Integer, false),
                new ColumnDeclaration("w500", ColumnKind.Integer, false),
                new ColumnDeclaration("w550", ColumnKind.Integer, false),
            ]),
            new InMemoryRowSource(["planned", "w500", "w550"], [["10", "3", "5"], ["4", "1", "3"]]));
        var step = new DistributionStep(["w500", "w550"], "planned", ordered: true, remainder: "other");
        var said = step.After(ColumnState.Of(table));
        var untouched = new DistributionStep(["w500", "w550"], "planned");

        step.MakeAnswers(table);
        untouched.MakeAnswers(table);

        Assert.Equal(
            said.Columns.Select(column => $"{column.Name}:{column.Kind}"),
            table.Columns.Select(column => $"{column.Name}:{column.Kind}"));
        Assert.Equal([0.3, 0.25], table.NumbersOf("w500"));
        Assert.Equal([0.2, 0.0], table.NumbersOf("other"));
        Assert.Same(ColumnState.None, untouched.After(ColumnState.None));
    }

    [Fact]
    public void AFlockServedLater_AwaitsItsBands_AndItsSharesComeBackAsBirdsOfThePlannedFlock()
    {
        var prepared = Planned(3);
        var handedIn = new InMemoryRowSource(["age", "planned"], [["35", "1000"]]);
        var served = prepared.Served(handedIn);

        var back = prepared.BackToOriginal([[0.2, 0.5, 0.25, 0.05]], served, handedIn);

        Assert.Equal(["age", "planned"], served.FeatureNames);
        Assert.Equal([200.0, 500.0, 250.0, 50.0], back[0].Select(birds => Math.Round(birds, 9)));
    }

    [Fact]
    public void WhatIsLeftOfTheWhole_IsMadeOnlyWithWhatTheWholeIs_AndUnderANameOfItsOwn()
    {
        var withoutWhole = Assert.Throws<ArgumentException>(() => new DistributionStep(["w500", "w550"], scaleBy: null, ordered: true, remainder: "other"));
        var aBand = Assert.Throws<ArgumentException>(() => new DistributionStep(["w500", "w550"], "chicks", ordered: true, remainder: "w550"));
        var theWhole = Assert.Throws<ArgumentException>(() => new DistributionStep(["w500", "w550"], "chicks", ordered: true, remainder: "chicks"));
        var inAFile = Assert.Throws<PipelineFileException>(() => StepCatalog.BuiltIn().ReadStep(
            """{"step": "target.distribution", "columns": ["w500", "w550"], "remainder": "other"}"""));

        Assert.Contains("'scaleBy'", withoutWhole.Message, StringComparison.Ordinal);
        Assert.Contains("'w550' is one of the shares", aBand.Message, StringComparison.Ordinal);
        Assert.Contains("'chicks'", theWhole.Message, StringComparison.Ordinal);
        Assert.Contains("'scaleBy'", inAFile.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADistributionWithWhatIsLeft_MakesItsAnswer_IsWrittenWithIt_AndReadBack()
    {
        var left = new DistributionStep(["w500", "w550"], "chicks", ordered: false, remainder: "other");
        var whole = new DistributionStep(["w500", "w550"], "chicks");
        var catalog = StepCatalog.BuiltIn();

        Assert.Equal("other", left.Remainder);
        Assert.Null(whole.Remainder);
        Assert.Equal(["w500", "w550", "other"], left.Answers);
        Assert.True(((INamesTheAnswer)left).MakesItsAnswer);
        Assert.False(((INamesTheAnswer)whole).MakesItsAnswer);
        Assert.True(left.Undoes("other"));
        Assert.Equal(left, PipelineDeclaration.FromJson(Declared(left), catalog).Output);
        Assert.Contains("\"remainder\":\"other\"", Written(left), StringComparison.Ordinal);
        Assert.DoesNotContain("remainder", Written(whole), StringComparison.Ordinal);
        Assert.DoesNotContain("remainder", catalog.Describe("target.distribution").Template, StringComparison.Ordinal);
        Assert.NotEqual(left, whole with { });
        Assert.NotEqual(left.GetHashCode(), whole.GetHashCode());
        Assert.Contains(left.After(ColumnState.None).Columns, column => column.Name == "other");
    }

    // A declaration around the output, written as a file writes it.
    private static string Declared(DistributionStep output) => new PipelineDeclaration(
    [
        new ReadCsvStep("flocks.csv"),
        new DeclareStep([.. new[] { "chicks", "w500", "w550" }.Select(name => new ColumnDeclaration(name, ColumnKind.Number, Optional: false))]),
        new SplitAtRandomStep(new SplitShares(0.50, 0, 0.50), 3),
        output,
    ]).ToJson();

    // The output as that file writes it, alone and with no spaces.
    private static string Written(DistributionStep output)
    {
        using var file = JsonDocument.Parse(Declared(output));

        return file.RootElement.GetProperty("declaration")[3].GetRawText().Replace(" ", string.Empty, StringComparison.Ordinal);
    }
}
