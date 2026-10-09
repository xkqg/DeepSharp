// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The rows a model is measured on at the very end must not move when the rows it learns from and chooses with are
/// dealt again. A split at random used to deal every part from one seed, so a second seed gave a second test part, and
/// a search that tried seeds to see how much a result depended on the luck of the deal had measured each trial on rows
/// another trial had learned from. A test seed fixes the test part and lets the seed deal the rest.
/// </summary>
public class SplitLockedTestTests
{
    private static PreparedData Divided(int seed, int? testSeed)
    {
        var declared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"));

        return (testSeed is { } locked ? declared.SplitAtRandom(0.70, 0.15, seed, locked) : declared.SplitAtRandom(0.70, 0.15, seed))
            .Build()
            .Run();
    }

    private static HashSet<RowKey> KeysIn(PreparedData prepared, Part part) =>
        [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Select(row => prepared.Table.Identities[row].Key)];

    [Fact]
    public void WithATestSeed_TheTestPartIsTheSameRows_WhateverTheSeedDealsTheRestWith()
    {
        var first = Divided(seed: 1, testSeed: 99);
        var second = Divided(seed: 2, testSeed: 99);
        var third = Divided(seed: 3, testSeed: 99);

        Assert.Equal(KeysIn(first, Part.Test), KeysIn(second, Part.Test));
        Assert.Equal(KeysIn(first, Part.Test), KeysIn(third, Part.Test));
        Assert.NotEmpty(KeysIn(first, Part.Test));
    }

    [Fact]
    public void WithATestSeed_TheSeedStillDealsTrainingAndValidationDifferently()
    {
        var first = Divided(seed: 1, testSeed: 99);
        var second = Divided(seed: 2, testSeed: 99);

        Assert.NotEqual(KeysIn(first, Part.Train), KeysIn(second, Part.Train));
        Assert.NotEqual(KeysIn(first, Part.Validation), KeysIn(second, Part.Validation));
    }

    [Fact]
    public void WithATestSeed_EveryRowIsInOnePartAndTheTestRowsAreNeverLearnedFrom()
    {
        var prepared = Divided(seed: 5, testSeed: 99);
        var test = KeysIn(prepared, Part.Test);

        Assert.DoesNotContain(prepared.Parts, part => part == Part.Gap);
        Assert.Empty(KeysIn(prepared, Part.Train).Intersect(test));
        Assert.Empty(KeysIn(prepared, Part.Validation).Intersect(test));
        Assert.Empty(KeysIn(prepared, Part.Train).Intersect(KeysIn(prepared, Part.Validation)));
    }

    [Fact]
    public void WithATestSeed_ThePartsHoldAboutTheSharesAsked()
    {
        var prepared = Divided(seed: 7, testSeed: 99);
        var rows = prepared.Table.RowCount;

        Assert.InRange(prepared.CountIn(Part.Train) / (double)rows, 0.68, 0.72);
        Assert.InRange(prepared.CountIn(Part.Validation) / (double)rows, 0.13, 0.17);
        Assert.InRange(prepared.CountIn(Part.Test) / (double)rows, 0.13, 0.17);
    }

    [Fact]
    public void ExactCopiesOfARow_NeverLandInTwoParts_WithATestSeedEither()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived"), Remainder.Keep)
            .SplitAtRandom(0.70, 0.15, 11, 99)
            .Build()
            .Run();

        var duplicates = prepared.Table.Duplicates(prepared.Parts);

        Assert.Equal(53, duplicates.Groups);
        Assert.Equal(0, duplicates.GroupsAcrossParts);
    }

    [Fact]
    public void ATestSeedEqualToTheSeed_DealsExactlyAsNoTestSeedDoes()
    {
        // Rows are ranked one by one by what they say and the seed, so ranking those the test seed left over, by the
        // same number, keeps the order they had: asking for the same number twice is asking for nothing more.
        var alone = Divided(seed: 42, testSeed: null);
        var same = Divided(seed: 42, testSeed: 42);

        Assert.Equal(alone.Parts, same.Parts);
    }

    [Fact]
    public void WithoutATestSeed_TheStepIsWhatItWas_AndWritesNoKeyForIt()
    {
        var step = new SplitAtRandomStep(SplitShares.Of(0.70, 0.15), 42);
        var written = JsonDocument.Parse(step.Canonical()).RootElement;

        Assert.Null(step.TestSeed);
        Assert.False(written.TryGetProperty("testSeed", out _));
        Assert.Equal(step, SplitAtRandomStep.ReadFrom(written));
    }

    [Fact]
    public void ATestSeed_IsWrittenInTheFile_AndReadBackEqual()
    {
        var step = new SplitAtRandomStep(SplitShares.Of(0.70, 0.15), 42, 99);
        var written = JsonDocument.Parse(step.Canonical()).RootElement;

        Assert.Equal(99, written.GetProperty("testSeed").GetInt32());
        Assert.Equal(99, SplitAtRandomStep.ReadFrom(written).TestSeed);
        Assert.Equal(step, SplitAtRandomStep.ReadFrom(written));
        Assert.NotEqual(step, new SplitAtRandomStep(SplitShares.Of(0.70, 0.15), 42));
    }

    [Fact]
    public void ATestSeedBelowNought_IsRefusedByName_WhetherTypedOrWrittenInAFile()
    {
        var typed = Assert.Throws<ArgumentOutOfRangeException>(() => new SplitAtRandomStep(SplitShares.Of(0.70, 0.15), 42, -5));

        Assert.Contains("testSeed", typed.Message, StringComparison.Ordinal);

        var step = new SplitAtRandomStep(SplitShares.Of(0.70, 0.15), 42, 99);
        var text = System.Text.Encoding.UTF8.GetString(step.Canonical()).Replace("\"testSeed\":99", "\"testSeed\":-1", StringComparison.Ordinal);
        var written = JsonDocument.Parse(text).RootElement;

        Assert.Contains("testSeed", Assert.Throws<ArgumentOutOfRangeException>(() => SplitAtRandomStep.ReadFrom(written)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePipelineFileOfALockedSplit_ReadsBackAndRunsTheSameDeal()
    {
        var declared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitAtRandom(0.70, 0.15, 3, 99)
            .Build();

        var again = PipelineDeclaration.FromJson(declared.Declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declared.Declaration, again);
        Assert.Equal(99, again.Steps.OfType<SplitAtRandomStep>().Single().TestSeed);
        Assert.Equal(3, again.Steps.OfType<SplitAtRandomStep>().Single().Seed);
    }

    [Fact]
    public void ADeclarationWithATestSeed_IsWrittenAsVersionEight()
    {
        // A key an older reader would refuse by name belongs to the version that refuses the file whole.
        var text = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitAtRandom(0.70, 0.15, 3, 99)
            .Build()
            .Declaration.ToJson();

        Assert.Equal(PipelineDeclaration.Version, JsonDocument.Parse(text).RootElement.GetProperty("version").GetInt32());
        Assert.Equal(8, PipelineDeclaration.Version);
    }
}
