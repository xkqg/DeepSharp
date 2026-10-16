// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The columns a column of words was written down as, one for each category, named by the encoder's own rule: each category the
/// training rows held in the order it learned them, the column kept for one they never held, and the one that marks a gap — what
/// a layer that reads a category from its block reads.
/// </summary>
public class CategoryBlockTests
{
    // Every category, and a gap, among the earliest rows, which the model learns from.
    private const string Rows = "x,c,y\n1,a,0\n2,b,1\n3,c,0\n4,,1\n5,a,0\n6,b,1\n";

    private static PreparedData Run(Func<FittingBuilder, FittingBuilder> encoded) =>
        encoded(Pdd.Create().Read(CsvRowSource.FromText(Rows), "rows").Declare(schema => schema.Integer("x").Number("y").Category("c")).SplitByTime("x", 0.8))
            .Target("y")
            .Build()
            .Run();

    [Fact]
    public void AColumnWrittenDownOneColumnPerCategory_HasItsCategoriesItsReserveAndItsMarker_InThatOrder()
    {
        Assert.Equal(["c_a", "c_b", "c_c", "c_other", "c_was_missing"], Run(fitting => fitting.EncodeCategories()).CategoryBlock("c"));
        Assert.Equal(["c_a", "c_b", "c_c", "c_other", "c_was_missing"], Run(fitting => fitting.Encode("c")).CategoryBlock("c"));
    }

    [Fact]
    public void AnEncoderThatRefusesAnUnseenCategory_KeepsNoReserve()
    {
        Assert.Equal(["c_a", "c_b", "c_c", "c_was_missing"], Run(fitting => fitting.Encode("c", unseen: Unseen.Refuse)).CategoryBlock("c"));
    }

    [Fact]
    public void AColumnWrittenDownAsItsPlaces_OrNotWrittenDownAtAll_HasNoBlock()
    {
        var places = Run(fitting => fitting.Encode("c", As.Ordinal));

        Assert.Contains("encode it with as: oneHot", Assert.Throws<InvalidOperationException>(() => places.CategoryBlock("c")).Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => Run(fitting => fitting.EncodeCategories()).CategoryBlock("x"));
        Assert.Throws<ArgumentException>(() => places.CategoryBlock(" "));
    }

    [Fact]
    public void EveryRowHoldsAOneInExactlyOneColumnOfItsBlock()
    {
        var prepared = Run(fitting => fitting.EncodeCategories());
        var block = prepared.CategoryBlock("c");

        for (var row = 0; row < prepared.Table.RowCount; row++)
        {
            Assert.Equal(1, block.Sum(name => prepared.Table.NumbersOf(name)[row]));
        }
    }
}
