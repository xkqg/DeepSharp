// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What each row holds here is digested once, however many things ask.
/// </summary>
/// <remarks>
/// The rows that are there more than once are found by the key each row holds where the step stands — its columns as
/// they are there, which is not the key the row was read with — so a profile beside a correlation, or two profiles at
/// one place, walked every cell of every column again for the same answer. Measured on the published passenger list, a
/// second profile at one place cost another 1.55 MB of the 1.88 MB a profile costs at all. The answer is unchanged: the
/// rows are what they are until a column arrives or leaves, so the keys are kept until one does.
/// </remarks>
public class RowKeyMemoTests
{
    [Fact]
    public void AskingASecondTimeWhatTheRowsHold_DigestsNothingAgain()
    {
        var counted = Counting(1.0, 2.0, 2.0);
        var table = new Table([counted]);

        _ = table.Duplicates();
        var once = counted.Reads;

        _ = table.Duplicates();
        _ = table.Duplicates();

        Assert.Equal(once, counted.Reads);
    }

    [Fact]
    public void AColumnPutIntoTheTable_MakesTheKeysAgain()
    {
        // The keys say what a row holds, so a column arriving is a different row and a different key.
        var counted = Counting(1.0, 2.0, 2.0);
        var table = new Table([counted]);

        _ = table.Duplicates();
        var once = counted.Reads;

        table.Put(new Column<double>("b", ColumnKind.Number, [1.0, 1.0, 1.0]));

        // a=[1,2,2] beside b=[1,1,1] makes the last two rows the same row.
        Assert.Equal(2, table.Duplicates().Rows);
        Assert.Equal(once + table.RowCount, counted.Reads);
    }

    [Fact]
    public void AColumnTakenAway_MakesTheKeysAgain()
    {
        var counted = Counting(1.0, 2.0, 3.0);
        var table = new Table([counted, new Column<double>("b", ColumnKind.Number, [1.0, 1.0, 1.0])]);

        _ = table.Duplicates();
        var once = counted.Reads;

        Assert.True(table.Remove("b"));
        _ = table.Duplicates();

        Assert.Equal(once + table.RowCount, counted.Reads);
    }

    [Fact]
    public void TheKeptKeys_AreTheKeysItWouldMakeAgain()
    {
        // Keeping them may not change the answer: the same rows come back as duplicates, and the table's own digest is
        // the digest it always was.
        var kept = new Table([new TextColumn("word", ["a", "b", "a", "c"])]);
        var fresh = new Table([new TextColumn("word", ["a", "b", "a", "c"])]);

        _ = kept.Duplicates();
        _ = kept.Duplicates();

        Assert.Equal(fresh.Duplicates(), kept.Duplicates());
        Assert.Equal(fresh.Digest(), kept.Digest());
    }

    [Fact]
    public void TheKeysAsTheyStandHere_AreNotTheKeysTheRowsWereReadWith()
    {
        // A table made from rows that carry their own identities keeps those for finding a row again; what it digests
        // for the rows that repeat is its own columns as they are here. Keeping one may never be read as the other.
        var read = new Table([new TextColumn("word", ["a", "b"])]);
        var here = new Table([new TextColumn("word", ["a", "b"]), new TextColumn("more", ["x", "x"])], read.Identities);

        Assert.Equal(read.Identities.Select(identity => identity.Key), here.Identities.Select(identity => identity.Key));
        Assert.NotEqual(read.Digest(), here.Duplicates().ToString());
        Assert.Equal(0, here.Duplicates().Rows);
    }

    private static CountingColumn Counting(params double[] values) =>
        new(new Column<double>("a", ColumnKind.Number, values.Select(value => (double?)value)));

    /// <summary>A column that counts how often its cells are read as text, to see how often the keys are made.</summary>
    private sealed class CountingColumn(IColumn column) : IColumn
    {
        public int Reads { get; private set; }

        public string Name => column.Name;

        public ColumnKind Kind => column.Kind;

        public int Count => column.Count;

        public bool IsMissing(int row) => column.IsMissing(row);

        public string? TextAt(int row)
        {
            Reads++;

            return column.TextAt(row);
        }

        public IColumn Rows(IReadOnlyList<int> rows) => new CountingColumn(column.Rows(rows));

        public int LeadingGaps() => column.LeadingGaps();
    }
}
