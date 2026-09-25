// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
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
}
