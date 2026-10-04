// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Shuffling the rows before they are divided.
/// </summary>
/// <remarks>
/// A file is often written in an order that means something — every survivor first, every month in turn — and a model
/// handed the rows that way can learn the order instead of the data. Shuffling before the split is what gives each part
/// the same mixture; shuffling again inside each epoch is the training loop's own, and it already does that.
/// Without this verb the rows keep the order they were read in.
/// </remarks>
public class ShuffleTests
{
    [Fact]
    public void WithoutIt_TheRowsKeepTheOrderTheyWereReadIn()
    {
        var prepared = Passengers(shuffled: false);

        Assert.Equal(
            Enumerable.Range(0, prepared.Table.RowCount),
            Enumerable.Range(0, prepared.Table.RowCount).Select(row => prepared.Table.Identities[row].ReadAt));
    }

    [Fact]
    public void WithIt_TheRowsStandInAnotherOrder_AndEveryRowIsStillThereExactlyOnce()
    {
        var prepared = Passengers(shuffled: true);
        var read = Enumerable.Range(0, prepared.Table.RowCount).Select(row => prepared.Table.Identities[row].ReadAt).ToArray();

        Assert.NotEqual(Enumerable.Range(0, read.Length), read);
        Assert.Equal(Enumerable.Range(0, read.Length), read.Order());
    }

    [Fact]
    public void TheSameSeed_ShufflesTheSameWay_AndAnotherSeedAnother()
    {
        static int[] Order(int seed) =>
            [.. Enumerable.Range(0, 891).Select(row => Passengers(shuffled: true, seed: seed).Table.Identities[row].ReadAt)];

        Assert.Equal(Order(20260929), Order(20260929));
        Assert.NotEqual(Order(20260929), Order(7));
    }

    [Fact]
    public void ItChangesNoValueAndNoColumn_OnlyWhereTheRowsStand()
    {
        var plain = Passengers(shuffled: false);
        var shuffled = Passengers(shuffled: true);

        Assert.Equal(plain.Table.RowCount, shuffled.Table.RowCount);
        Assert.Equal(plain.Table.Columns.Select(column => column.Name), shuffled.Table.Columns.Select(column => column.Name));
        Assert.Equal(plain.Table.Digest(), shuffled.Table.Digest());
    }

    [Fact]
    public void ItIsTheOneThingThatSaysTheOrder_SoItIsNotWrittenBesideAnOrderBy()
    {
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .OrderBy("Date")
            .Shuffle()
            .SplitByTime("Date", 0.70, 0.15)
            .Target("AAPL.Close")
            .Build());

        Assert.Contains("order", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ItIsReadBackFromItsFile_AsItWroteItself()
    {
        var read = (ShuffleStep)Shipped.Catalog().ReadStep("""{"step": "shuffle", "seed": 7}""");

        Assert.Equal(7, read.Seed);
        Assert.Equal("shuffle", read.Verb);
        Assert.Equal(7, ShuffleStep.Since);
    }

    private static PreparedData Passengers(bool shuffled, int seed = 20260929)
    {
        var chain = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Number("fare"));

        return (shuffled ? chain.Shuffle(seed) : chain)
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Build()
            .Run();
    }
}
