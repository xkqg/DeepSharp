// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A column worked out from another carries that column's gaps, and the refusal says so.
/// </summary>
/// <remarks>
/// The features are added above the split and the gaps are filled below it, because what fills a gap is learned from
/// the training rows alone. So a sum of two columns, one of which has a gap, is itself a gap — and filling the column
/// it was made from afterwards does not reach back into it. Whoever meets that at the handover should be told which
/// column it came from and what to do, rather than being told only that a gap is a gap.
/// </remarks>
public class DerivedGapTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    [Fact]
    public void AGapInAColumnAFeatureIsMadeFrom_IsRefusedWithTheColumnItCameFrom()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived", "sibsp", "parch")
                .Optional("age", ColumnKind.Number))
            .AddFeature("older", "age", Arithmetic.Plus, "sibsp")
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing(fill => fill.Median("age"))
            .Target("survived")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Train));

        Assert.Contains("'older'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'age'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("worked out from", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AGapInAColumnNothingMade_IsRefusedAsItAlwaysWas()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived", "sibsp")
                .Optional("age", ColumnKind.Number))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Target("survived")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Batch(Part.Train));

        Assert.Contains("'age' is still a gap", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("worked out from", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFeatureWhoseColumnsAreFilledAboveIt_HandsOverWithoutAWord()
    {
        // The way to have it: the rows whose gaps nothing can answer are dropped above the split, where dropping does
        // not change the shares a split promised.
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived", "sibsp", "parch")
                .Optional("age", ColumnKind.Number))
            .DropGaps("age")
            .AddFeature("older", "age", Arithmetic.Plus, "sibsp")
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Normalise("age", "older", "sibsp", "parch")
            .Target("survived")
            .Build()
            .Run();

        Assert.NotEmpty(prepared.Batch(Part.Train).Features);
    }
}
