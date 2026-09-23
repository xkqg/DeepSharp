// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Every row carries one identity, and the table owns it: where the row was when it was read, and a key made
/// from what the row says. The place is what a run uses to find a row again — the way back for a target, a
/// row handed in to be served — and the key is what stays the same when the same rows arrive in another order
/// tomorrow. A walk that kept its own list of where rows came from beside the table was two lists changed by
/// different acts, and they drift.
/// </summary>
public class RowIdentityTests
{
    private static Table Bound(string csv, params ColumnDeclaration[] columns) =>
        SchemaBinding.Bind(new DeclareStep(columns), CsvRowSource.FromText(csv));

    private static ColumnDeclaration Number(string name) => new(name, ColumnKind.Number, false);

    [Fact]
    public void AKeyIsMadeFromWhatTheRecordSays_NotFromTheOrderItsColumnsCameIn()
    {
        var one = RowKey.Of(["a", "b"], ["1", "x"]);
        var swapped = RowKey.Of(["b", "a"], ["x", "1"]);

        Assert.Equal(one, swapped);
        Assert.NotEqual(one, RowKey.Of(["a", "b"], ["1", "y"]));
        Assert.NotEqual(one, RowKey.Of(["a", "c"], ["1", "x"]));
    }

    [Fact]
    public void AGapAndAnEmptyCell_AreDifferentRows_ButSpacesAroundAWordAreNot()
    {
        Assert.NotEqual(RowKey.Of(["a"], [null]), RowKey.Of(["a"], [string.Empty]));
        Assert.Equal(RowKey.Of(["a"], ["3"]), RowKey.Of(["a"], [" 3 "]));
    }

    [Fact]
    public void AKeyIsTheSameOnEveryMachine()
    {
        // SHA-256 over a written-down encoding, and never a hash code: a split that ranks rows by their keys
        // has to put a row in the same part on every machine and in every process, this year and next. If this
        // number changes, every saved split has silently changed with it.
        Assert.Equal(
            "0a5cc1133acc91d13a0205fd34510a5cd790d03d560573c857de048ce5d8cdac",
            RowKey.Of(["name", "age", "cabin"], ["Braund, Mr. Owen Harris", "22", null]).ToString());
    }

    [Fact]
    public void ARowIsKeyedByTheWholeRecord_SoLeavingAColumnOutOfTheSchemaChangesNoKey()
    {
        // The key comes from the record as it arrived, every column of it, rather than from the columns the
        // schema kept: taking a column out of a notebook's schema must not move a row to another part.
        const string csv = "a,b,c\n1,2,x\n3,4,y\n";

        var wide = Bound(csv, Number("a"), Number("b"));
        var narrow = Bound(csv, Number("a"));

        Assert.Equal(wide.Identities, narrow.Identities);
        Assert.Equal(RowKey.Of(["a", "b", "c"], ["1", "2", "x"]), narrow.Identities[0].Key);
    }

    [Fact]
    public void EveryRowRemembersWhereItWasRead()
    {
        var table = Bound("a\n5\n6\n7\n", Number("a"));

        Assert.Equal([0, 1, 2], table.Identities.Select(identity => identity.ReadAt));
    }

    [Fact]
    public void KeepingRowsAndPuttingThemInOrder_TakeTheirIdentitiesWithThem()
    {
        var table = Bound("a\n5\n6\n7\n8\n", Number("a"));

        var kept = table.Keep([true, false, true, true]);
        var ordered = kept.Ordered([2, 0, 1]);

        Assert.Equal([0, 2, 3], kept.Identities.Select(identity => identity.ReadAt));
        Assert.Equal([5.0, 7.0, 8.0], kept.NumbersOf("a").Select(value => value!.Value));
        Assert.Equal([3, 0, 2], ordered.Identities.Select(identity => identity.ReadAt));
        Assert.Equal([8.0, 5.0, 7.0], ordered.NumbersOf("a").Select(value => value!.Value));
        Assert.Equal(table.Identities[3], ordered.Identities[0]);
    }

    [Fact]
    public void AColumnOfWordsMovesTheSameWay()
    {
        var table = new Table([new TextColumn("a", ColumnKind.Category, [null, "x", "y"])]);

        var kept = table.Keep([false, true, true]).Ordered([1, 0]);

        Assert.Equal(["y", "x"], [kept["a"].TextAt(0), kept["a"].TextAt(1)]);
        Assert.Equal(ColumnKind.Category, kept["a"].Kind);
    }

    [Fact]
    public void AMaskOrAnOrderThatDoesNotFitTheRows_IsRefused()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0, 2.0])]);

        Assert.Throws<ArgumentException>(() => table.Keep([true]));
        Assert.Throws<ArgumentException>(() => table.Ordered([0]));
        Assert.Throws<ArgumentException>(() => table.Ordered([0, 0]));
        Assert.Throws<ArgumentException>(() => table.Ordered([0, 2]));
        Assert.Throws<ArgumentNullException>(() => table.Keep(null!));
        Assert.Throws<ArgumentNullException>(() => table.Ordered(null!));
    }

    [Fact]
    public void ATableBuiltByHand_KeysItsRowsFromItsOwnCells_ByTheSameRule()
    {
        var table = new Table([
            new Column<double>("fare", ColumnKind.Number, [7.25, null]),
            new TextColumn("sex", ["male", "female"]),
        ]);

        Assert.Equal(RowKey.Of(["fare", "sex"], ["7.25", "male"]), table.Identities[0].Key);
        Assert.Equal(RowKey.Of(["fare", "sex"], [null, "female"]), table.Identities[1].Key);
        Assert.Equal([0, 1], table.Identities.Select(identity => identity.ReadAt));
    }

    [Fact]
    public void ATableGivenIdentities_KeepsThem_AndRefusesTheWrongNumber()
    {
        var identity = new RowIdentity(7, RowKey.Of(["a"], ["1"]));
        var columns = new[] { new Column<double>("a", ColumnKind.Number, [1.0]) };

        Assert.Equal(identity, Assert.Single(new Table(columns, [identity]).Identities));
        Assert.Throws<ArgumentException>(() => new Table(columns, [identity, identity]));
    }

    [Fact]
    public void KeysSortTheSameWayEverywhere_AndSayWhatTheyAre()
    {
        var one = RowKey.Of(["a"], ["1"]);
        var other = RowKey.Of(["a"], ["2"]);

        Assert.Equal(-other.CompareTo(one), one.CompareTo(other));
        Assert.Equal(0, one.CompareTo(one));
        Assert.Equal(64, one.ToString().Length);
        Assert.Throws<ArgumentNullException>(() => RowKey.Of(null!, []));
        Assert.Throws<ArgumentException>(() => RowKey.Of(["a"], []));
    }
}
