// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A split that cuts more than once. Every split here cut once, so a small table was measured by the luck of one deal and a
/// series was not measured the way forecasting is, by moving the cut forward and fitting again. A fold is the i-th of k slices of
/// the rows that are neither test nor predict, dealt by the seed: the slice is validation, the rest is training, the test rows are
/// the same on every fold, and every fold is an ordinary pipeline of its own with a fit of its own. A rolling origin does it for a
/// series: the training rows grow, validation is the block after them, and the test rows are one block at the end.
/// </summary>
public class FoldSplitTests
{
    private const int Folds = 5;

    private static PreparedData Dealt(int index, int? testSeed = null, int seed = 42)
    {
        var declared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"));

        return (testSeed is { } locked
                ? declared.SplitAtRandom(0.80, new Fold(Folds, index), seed, locked)
                : declared.SplitAtRandom(0.80, new Fold(Folds, index), seed))
            .Build()
            .Run();
    }

    private static HashSet<RowKey> KeysIn(PreparedData prepared, Part part) =>
        [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Select(row => prepared.Table.Identities[row].Key)];

    [Fact]
    public void TheFolds_AreDisjoint_AndTogetherTheRowsThatAreNeitherTestNorPredict()
    {
        var all = Enumerable.Range(0, Folds).Select(index => Dealt(index)).ToArray();
        var learnable = KeysIn(all[0], Part.Train).Union(KeysIn(all[0], Part.Validation)).ToHashSet();

        for (var one = 0; one < Folds; one++)
        {
            Assert.Equal(learnable, KeysIn(all[one], Part.Train).Union(KeysIn(all[one], Part.Validation)).ToHashSet());
            Assert.Empty(KeysIn(all[one], Part.Train).Intersect(KeysIn(all[one], Part.Validation)));

            for (var other = one + 1; other < Folds; other++)
            {
                Assert.Empty(KeysIn(all[one], Part.Validation).Intersect(KeysIn(all[other], Part.Validation)));
            }
        }

        Assert.Equal(learnable, all.SelectMany(each => KeysIn(each, Part.Validation)).ToHashSet());
        Assert.DoesNotContain(all, each => each.Parts.Contains(Part.Gap));
    }

    [Fact]
    public void EachFold_HoldsAboutAFifthOfTheRowsAModelLearnsFromAndChoosesWith()
    {
        for (var index = 0; index < Folds; index++)
        {
            var prepared = Dealt(index);
            var learnable = prepared.CountIn(Part.Train) + prepared.CountIn(Part.Validation);

            Assert.InRange(prepared.CountIn(Part.Validation) / (double)learnable, 0.17, 0.23);
        }
    }

    [Theory]
    [InlineData(null)]
    [InlineData(99)]
    public void TheTestRows_AreTheSameOnEveryFold_WithATestSeedAndWithout(int? testSeed)
    {
        var first = KeysIn(Dealt(0, testSeed), Part.Test);

        Assert.NotEmpty(first);

        for (var index = 1; index < Folds; index++)
        {
            Assert.Equal(first, KeysIn(Dealt(index, testSeed), Part.Test));
        }
    }

    [Fact]
    public void ARowThatIsRepeated_NeverLandsInTwoFolds_NorOnBothSidesOfOne()
    {
        for (var index = 0; index < Folds; index++)
        {
            var prepared = Pdd.Create()
                .ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived"), Remainder.Keep)
                .SplitAtRandom(0.80, new Fold(Folds, index), 42, 99)
                .Build()
                .Run();
            var duplicates = prepared.Table.Duplicates(prepared.Parts);

            Assert.Equal(53, duplicates.Groups);
            Assert.Equal(0, duplicates.GroupsAcrossParts);
        }
    }

    [Fact]
    public void ARecordRepeatedAcrossASliceBoundary_StaysInOneSlice()
    {
        // Eight records, each four times over, dealt into five slices of about five rows: a boundary falls inside a record's copies,
        // and the copies all stay on one side of it.
        var rows = Enumerable.Range(0, 32).Select(row => (IReadOnlyList<string?>)[(row % 8).ToString(CultureInfo.InvariantCulture), ((row % 8) * 7).ToString(CultureInfo.InvariantCulture)]).ToList();

        for (var index = 0; index < Folds; index++)
        {
            var prepared = Pdd.Create()
                .Read(new InMemoryRowSource(["a", "b"], rows), "eight records, four times each")
                .Declare(schema => schema.Integer("a", "b"))
                .SplitAtRandom(0.75, new Fold(Folds, index), 42)
                .Build()
                .Run();
            var duplicates = prepared.Table.Duplicates(prepared.Parts);

            Assert.Equal(8, duplicates.Groups);
            Assert.Equal(0, duplicates.GroupsAcrossParts);
            Assert.NotEmpty(KeysIn(prepared, Part.Validation));
        }
    }

    [Fact]
    public void AFold_IsAFitOfItsOwn_SoANetworkTrainedBehindOneIsRefusedBesideAnother()
    {
        var digests = Enumerable.Range(0, Folds).Select(index => PipelineText.Of(Dealt(index)).Digest).ToArray();

        Assert.Equal(Folds, digests.Distinct().Count());
        Assert.NotEqual(Dealt(0).Declaration.ToJson(), Dealt(1).Declaration.ToJson());
    }

    [Fact]
    public void TheSameFold_IsDealtTheSameWayTwice_AndAnotherSeedDealsItOtherwise()
    {
        Assert.Equal(Dealt(2).Parts, Dealt(2).Parts);
        Assert.NotEqual(KeysIn(Dealt(2, seed: 42), Part.Validation), KeysIn(Dealt(2, seed: 43), Part.Validation));
    }

    [Fact]
    public void StratifiedFolds_KeepTheMixtureOfTheColumnInEveryFold()
    {
        for (var index = 0; index < Folds; index++)
        {
            var prepared = Pdd.Create()
                .ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived", "pclass"))
                .SplitStratified("survived", 0.80, new Fold(Folds, index), 42)
                .Build()
                .Run();
            var survived = prepared.Table["survived"];
            double Share(Part part)
            {
                var rows = Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).ToArray();

                return rows.Count(row => survived.TextAt(row) == "1") / (double)rows.Length;
            }

            Assert.InRange(Share(Part.Validation), Share(Part.Train) - 0.03, Share(Part.Train) + 0.03);
            Assert.InRange(Share(Part.Test), Share(Part.Train) - 0.04, Share(Part.Train) + 0.04);
        }
    }

    [Fact]
    public void WithoutAFold_TheStepIsWhatItWas_AndWritesNoKeyForOne()
    {
        var step = new SplitAtRandomStep(SplitShares.Of(0.70, 0.15), 42, 99);
        var written = JsonDocument.Parse(step.Canonical()).RootElement;

        Assert.Null(step.Fold);
        Assert.False(written.TryGetProperty("folds", out _));
        Assert.False(written.TryGetProperty("fold", out _));
        Assert.Equal(step, SplitAtRandomStep.ReadFrom(written));
    }

    [Fact]
    public void AFold_IsWrittenInTheFile_AndReadBackEqual_InBothSplitsAtRandom()
    {
        var random = new SplitAtRandomStep(SplitShares.Of(0.8), 42, null, new Fold(5, 2));
        var stratified = new SplitStratifiedStep("survived", SplitShares.Of(0.8), 42, new Fold(4, 3));
        var written = JsonDocument.Parse(random.Canonical()).RootElement;

        Assert.Equal(5, written.GetProperty("folds").GetInt32());
        Assert.Equal(2, written.GetProperty("fold").GetInt32());
        Assert.Equal(random, SplitAtRandomStep.ReadFrom(written));
        Assert.Equal(stratified, SplitStratifiedStep.ReadFrom(JsonDocument.Parse(stratified.Canonical()).RootElement));
    }

    [Theory]
    [InlineData(1, 0)]
    [InlineData(0, 0)]
    [InlineData(5, 5)]
    [InlineData(5, -1)]
    public void AFoldThatIsNotTheIthOfAtLeastTwo_IsRefused(int folds, int index) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new Fold(folds, index));

    [Fact]
    public void AValidationShareBesideFolds_IsNoUseToThem_ItOnlyAddsToTheRowsThatAreCut()
    {
        // Eighty per cent cut, as a train of 0.65 and a validation of 0.15 say it, is eighty per cent cut: the fold is validation.
        var table = Dealt(0).Table;
        Part[] said = new SplitAtRandomStep(SplitShares.Of(0.80), 42, null, new Fold(5, 2)).Assign(table);
        Part[] added = new SplitAtRandomStep(SplitShares.Of(0.65, 0.15), 42, null, new Fold(5, 2)).Assign(table);
        Part[] stratifiedSaid = new SplitStratifiedStep("survived", SplitShares.Of(0.80), 42, new Fold(5, 2)).Assign(table);
        Part[] stratifiedAdded = new SplitStratifiedStep("survived", SplitShares.Of(0.65, 0.15), 42, new Fold(5, 2)).Assign(table);
        Part[] sparedSaid = new SplitAtRandomStep(SplitShares.Of(0.80), 42, 7, new Fold(5, 2)).Assign(table);
        Part[] sparedAdded = new SplitAtRandomStep(SplitShares.Of(0.65, 0.15), 42, 7, new Fold(5, 2)).Assign(table);

        Assert.Equal(said, added);
        // Each group rounds its own shares to rows, so the cut may differ by a row in a group.
        Assert.InRange(stratifiedAdded.Count(part => part is Part.Train or Part.Validation), stratifiedSaid.Count(part => part is Part.Train or Part.Validation) - 2, stratifiedSaid.Count(part => part is Part.Train or Part.Validation) + 2);
        Assert.Contains(Part.Validation, stratifiedAdded);
        Assert.Equal(sparedSaid, sparedAdded);
        Assert.Contains(Part.Validation, added);
    }

    [Theory]
    [InlineData("""{"step":"split.atRandom","train":0.8,"validation":0,"test":0.2,"predict":0,"seed":42,"folds":5}""", "fold")]
    [InlineData("""{"step":"split.atRandom","train":0.8,"validation":0,"test":0.2,"predict":0,"seed":42,"fold":2}""", "folds")]
    [InlineData("""{"step":"split.atRandom","train":0.8,"validation":0,"test":0.2,"predict":0,"seed":42,"folds":5,"fold":5}""", "fold")]
    public void AFileThatSaysFoldsBadly_IsRefusedWhereItIsRead(string step, string names)
    {
        var json = $$"""{"version": 12, "declaration": [{"step": "read.csv", "path": "x.csv"}, {{step}}]}""";
        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        var said = string.Join(' ', refused.Faults.Select(fault => fault.Message));

        Assert.True(said.Contains(names, StringComparison.Ordinal), said);
    }

    // ---- a rolling origin: the cut moved forward, the test rows one block at the end ----

    private static PreparedData Rolled(int origin, int gap = 0, int origins = 3)
    {
        var rows = Enumerable.Range(0, 60).Select(day => (IReadOnlyList<string?>)
            [new DateTime(2024, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), (100 + (day * 3 % 17)).ToString(CultureInfo.InvariantCulture)]).ToList();

        return Pdd.Create()
            .Read(new InMemoryRowSource(["Date", "close"], rows), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("close"))
            .OrderBy("Date")
            .SplitByTime("Date", 0.75, new Origin(origins, origin), gap)
            .Build()
            .Run();
    }

    private static int[] DaysIn(PreparedData prepared, Part part) =>
        [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Select(row => (int)(((Column<DateTime>)prepared.Table["Date"])[row]!.Value - new DateTime(2024, 1, 1)).TotalDays).Order()];

    [Fact]
    public void ARollingOrigin_GrowsTheTrainingRows_PutsValidationRightAfterThem_AndKeepsOneFinalBlockForTest()
    {
        var test = DaysIn(Rolled(0), Part.Test);
        var trained = new List<int>();

        Assert.Equal(Enumerable.Range(45, 15), test);

        for (var origin = 0; origin < 3; origin++)
        {
            var prepared = Rolled(origin);
            var train = DaysIn(prepared, Part.Train);
            var validation = DaysIn(prepared, Part.Validation);

            Assert.Equal(test, DaysIn(prepared, Part.Test));
            Assert.NotEmpty(train);
            Assert.NotEmpty(validation);
            Assert.True(train.Max() < validation.Min(), "validation follows the training rows in time");
            Assert.Equal(Enumerable.Range(train.Min(), train.Length), train);
            Assert.Equal(Enumerable.Range(validation.Min(), validation.Length), validation);
            Assert.InRange(validation.Length, 11, 12);
            trained.Add(train.Length);
        }

        Assert.Equal(trained.Order(), trained);
        Assert.Equal(trained.Distinct().Count(), trained.Count);
    }

    [Fact]
    public void ARollingOrigin_LeavesTheRowsAfterValidationOutOfItsRun_AsRowsFittedOnByNothing()
    {
        var prepared = Rolled(0);
        var left = DaysIn(prepared, Part.Gap);

        Assert.NotEmpty(left);
        Assert.True(left.Min() > DaysIn(prepared, Part.Validation).Max());
        Assert.True(left.Max() < DaysIn(prepared, Part.Test).Min());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void ARollingOrigin_KeepsTheGapAtTheEndOfTrainingAndOfValidation_EveryOrigin(int origin)
    {
        var plain = Rolled(origin);
        var gapped = Rolled(origin, gap: 2);
        var train = DaysIn(plain, Part.Train);
        var validation = DaysIn(plain, Part.Validation);

        Assert.Equal(train[..^2], DaysIn(gapped, Part.Train));
        Assert.Equal(validation[..^2], DaysIn(gapped, Part.Validation));
        Assert.Equal(DaysIn(plain, Part.Test)[..^2], DaysIn(gapped, Part.Test));
    }

    [Fact]
    public void WithoutAnOrigin_TheStepIsWhatItWas_AndWritesNoKeyForOne()
    {
        var step = new SplitByTimeStep("Date", SplitShares.Of(0.7, 0.15), 1);
        var written = JsonDocument.Parse(step.Canonical()).RootElement;

        Assert.Null(step.Origin);
        Assert.False(written.TryGetProperty("origins", out _));
        Assert.False(written.TryGetProperty("origin", out _));
        Assert.Equal(step, SplitByTimeStep.ReadFrom(written));
    }

    [Fact]
    public void AnOrigin_IsWrittenInTheFile_ReadBackEqual_AndAValidationShareBesideItOnlyAddsToTheRowsThatAreCut()
    {
        var step = new SplitByTimeStep("Date", SplitShares.Of(0.75), 2, new Origin(3, 1));
        var written = JsonDocument.Parse(step.Canonical()).RootElement;

        Assert.Equal(3, written.GetProperty("origins").GetInt32());
        Assert.Equal(1, written.GetProperty("origin").GetInt32());
        Assert.Equal(step, SplitByTimeStep.ReadFrom(written));
        var table = Rolled(0).Table;

        Assert.Equal(
            new SplitByTimeStep("Date", SplitShares.Of(0.75), 0, new Origin(3, 1)).Assign(table),
            new SplitByTimeStep("Date", SplitShares.Of(0.70, 0.05), 0, new Origin(3, 1)).Assign(table));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Origin(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Origin(3, 3));
    }

    [Theory]
    [InlineData("origin", "origins")]
    [InlineData("origins", "origin")]
    public void AFileThatNamesOneKeyOfAnOrigin_WithoutTheOther_IsRefusedNamingWhatItSays(string left, string said)
    {
        var node = System.Text.Json.Nodes.JsonNode.Parse(new SplitByTimeStep("Date", SplitShares.Of(0.75), 2, new Origin(3, 1)).Canonical())!.AsObject();

        node.Remove(left);

        var refused = Assert.Throws<FormatException>(() => SplitByTimeStep.ReadFrom(JsonDocument.Parse(node.ToJsonString()).RootElement));

        Assert.Contains($"this file says only '{said}'", refused.Message, StringComparison.Ordinal);
    }

}
