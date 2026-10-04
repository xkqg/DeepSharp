// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The column that says where the gaps were is named in one place and written in one place.
/// </summary>
/// <remarks>
/// Three verbs write it — filling a gap, settling one, and encoding a column whose cell was empty — and each said the
/// name for itself, in its own words, which is the same sentence written three times. The name a reader sees does not
/// change: these are the columns already in every file written since the verbs existed.
/// </remarks>
public class MarkerTests
{
    [Theory]
    [InlineData("age")]
    [InlineData("a/b")]
    [InlineData("")]
    public void ThreeVerbsThatMarkAColumn_NameTheMarkerTheSameWay(string column)
    {
        var settled = new SettleGapsStep(column.Length == 0 ? "a" : column, With.Zero).MarkerColumn;

        Assert.Equal($"{(column.Length == 0 ? "a" : column)}_was_missing", settled);
        Assert.Equal(FillMissingStep.Of(column.Length == 0 ? "a" : column, With.Median).MarkerColumn, settled);
        Assert.Equal(new EncodeStep(column.Length == 0 ? "a" : column).MarkerColumn, settled);
    }

    [Fact]
    public void AMarkerPutOnATable_SaysOnePerRowWhereTheGapWas_AndStandsLast()
    {
        var table = new Table([
            new Column<double>("a", ColumnKind.Number, [1, null, 3]),
            new Column<double>("b", ColumnKind.Number, [1, 2, 3]),
        ]);

        table.Marks("a");

        Assert.Equal([0, 1, 0], table.NumbersOf("a_was_missing").Select(value => value!.Value));
        Assert.Equal(["a", "b", "a_was_missing"], table.Columns.Select(column => column.Name));
    }

    [Fact]
    public void AMarkerOnAColumnWithNoGaps_SaysSoForEveryRow()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1, 2, 3])]);

        table.Marks("a");

        Assert.Equal([0, 0, 0], table.NumbersOf("a_was_missing").Select(value => value!.Value));
    }
}
