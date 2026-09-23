// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A split decides which rows a model learns from and which it is measured on, so it has to decide the same
/// way every time it meets the same rows. It used to shuffle their places: the same file reversed shared only
/// 419 of its 624 training rows with itself, two copies of one passenger landed in training and test, and a
/// day split in time could end in training and start again in validation. Each row is now placed by what it
/// says rather than by where it stands.
/// </summary>
public class SplitKeyTests
{
    private static InMemoryRowSource Rows(string file, bool reversed)
    {
        var source = new CsvRowSource(Repository.Data(file));

        return new InMemoryRowSource(source.ColumnNames, reversed ? source.Rows.Reverse() : source.Rows);
    }

    private static PreparedData Divided(string split, bool reversed) => split switch
    {
        "byTime" => Pdd.Create()
            .Read(Rows("apple.csv", reversed), "apple prices")
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .SplitByTime("Date", 0.70, 0.15)
            .Build()
            .Run(),
        "atRandom" => Pdd.Create()
            .Read(Rows("titanic.csv", reversed), "passengers")
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitAtRandom(0.70, 0.15, seed: 42)
            .Build()
            .Run(),
        _ => Pdd.Create()
            .Read(Rows("titanic.csv", reversed), "passengers")
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitStratified("survived", 0.70, 0.15, seed: 42)
            .Build()
            .Run(),
    };

    private static HashSet<RowKey> KeysIn(PreparedData prepared, Part part) =>
        [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Select(row => prepared.Table.Identities[row].Key)];

    [Theory]
    [InlineData("atRandom")]
    [InlineData("stratified")]
    [InlineData("byTime")]
    public void EverySplit_PlacesEveryRowTheSame_WhateverOrderTheRowsArrive(string split)
    {
        var inOrder = Divided(split, reversed: false);
        var reversed = Divided(split, reversed: true);

        foreach (var part in new[] { Part.Train, Part.Validation, Part.Test })
        {
            Assert.Equal(KeysIn(inOrder, part), KeysIn(reversed, part));
        }
    }

    [Fact]
    public void ExactCopiesOfARow_NeverLandInTwoParts()
    {
        // Fifty-three passengers are in the published file more than once, and a model that learns one copy
        // and is measured on the other is measured on a row it has seen.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived"), Remainder.Keep)
            .SplitStratified("survived", 0.70, 0.15)
            .Build()
            .Run();

        var duplicates = prepared.Table.Duplicates(prepared.Parts);

        Assert.Equal(53, duplicates.Groups);
        Assert.Equal(0, duplicates.GroupsAcrossParts);
    }

    [Fact]
    public void AMomentIsNeverDividedBetweenTwoParts()
    {
        // Twenty-one days, two rows each: what happened at one moment is on one side of the line or the other.
        var rows = new InMemoryRowSource(
            ["Date", "value"],
            [
                .. Enumerable.Range(0, 42).Select(at => (IReadOnlyList<string?>)
                [
                    new DateTime(2020, 1, 1).AddDays(at / 2).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                    at.ToString(CultureInfo.InvariantCulture),
                ]),
            ]);

        var prepared = Pdd.Create()
            .Read(rows, "two rows a day")
            .Declare(schema => schema.Timestamp("Date").Number("value"))
            .SplitByTime("Date", 0.70, 0.15)
            .Build()
            .Run();

        var dates = (Column<DateTime>)prepared.Table["Date"];

        var divided = Enumerable.Range(0, prepared.Table.RowCount)
            .GroupBy(row => dates[row])
            .Count(day => day.Select(row => prepared.Parts[row]).Distinct().Count() > 1);

        Assert.Equal(0, divided);
    }

    [Fact]
    public void LeavingADeclaredColumnOut_MovesNoRowToAnotherPart()
    {
        // A row is placed by the record it was read from, not by the columns the schema kept, so taking a
        // column out of a notebook's schema cannot reshuffle the rows a model is measured on.
        PreparedData Split(bool withFare) => Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema =>
            {
                schema.Integer("survived");

                if (withFare)
                {
                    schema.Number("fare");
                }
            })
            .SplitAtRandom(0.70, 0.15, seed: 3)
            .Build()
            .Run();

        Assert.Equal(Split(withFare: true).Parts, Split(withFare: false).Parts);
    }

    [Fact]
    public void AnotherSeed_DealsTheRowsAnotherWay()
    {
        PreparedData Split(int seed) => Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived"))
            .SplitAtRandom(0.70, 0.15, seed)
            .Build()
            .Run();

        Assert.NotEqual(Split(1).Parts, Split(2).Parts);
        Assert.Equal(Split(1).Parts, Split(1).Parts);
    }
}
