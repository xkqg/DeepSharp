// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Rows that are the same row twice. A split divides rows, so two copies of one passenger can land one in
/// training and one in test — and a model then meets in test a row it has already learned, which reads as
/// skill. Nothing checked for it: only a column declared twice was refused, never a row that is there twice.
/// </summary>
public class DuplicateRowTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    [Fact]
    public void TitanicHoldsFiftyThreeGroupsOfIdenticalRows_NoneOfThemAcrossParts()
    {
        // Measured against the published file, every column taken as it is written: 53 groups, 160 rows in
        // them, so 107 copies beyond the first of each. A split that dealt rows by their places put 31 of those
        // groups in two parts at once; dealt by what each row says, every copy lands where its first copy does.
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived"), Remainder.Keep)
            .SplitStratified("survived", 0.70, 0.15)
            .Build()
            .Run();

        var duplicates = prepared.Table.Duplicates(prepared.Parts);

        Assert.Equal(53, duplicates.Groups);
        Assert.Equal(160, duplicates.Rows);
        Assert.Equal(107, duplicates.ExtraCopies);
        Assert.Equal(0, duplicates.GroupsAcrossParts);
    }

    [Fact]
    public void AGapIsPartOfWhatARowSays()
    {
        // Two rows that differ only in where one of them has a gap are two rows, not one: absent is a value
        // of its own everywhere in this library, and here too.
        var table = new Table([
            new Column<double>("a", ColumnKind.Number, [1.0, 1.0, null, 2.0]),
            new TextColumn("b", ["x", "x", "x", "x"]),
        ]);

        var duplicates = table.Duplicates();

        Assert.Equal(1, duplicates.Groups);
        Assert.Equal(2, duplicates.Rows);
        Assert.Equal(1, duplicates.ExtraCopies);
        Assert.Equal(0, duplicates.GroupsAcrossParts);
    }

    [Fact]
    public void ATableWithoutRepeats_HasNone()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0, 2.0, 3.0])]);

        Assert.Equal(default, table.Duplicates([Part.Train, Part.Test, Part.Train]));
        Assert.Equal(default, new Table([]).Duplicates());
    }

    [Fact]
    public void PartsThatDoNotMatchTheRows_AreRefused()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0, 1.0])]);

        Assert.Throws<ArgumentException>(() => table.Duplicates([Part.Train]));
    }
}
