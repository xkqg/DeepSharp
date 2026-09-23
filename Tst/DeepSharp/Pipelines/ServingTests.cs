// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The handover serves as well as trains. A row that arrives a year from now is replayed through the saved
/// pipeline and handed over in exactly the shape the training rows were — the same columns in the same order,
/// refused for the same reasons — without an answer, since the answer is what is being asked. And because a
/// replay can drop rows, each served row says which of the handed-in rows it is.
/// </summary>
public class ServingTests
{
    private static PreparedData Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("age")
            .EncodeCategories()
            .Target("survived")
            .Build()
            .Run();

    [Fact]
    public void ServedRows_ComeOutInTheShapeTheTrainingRowsDid_WithoutAnAnswer()
    {
        var trained = Passengers();

        var served = trained.Served(new InMemoryRowSource(
            ["pclass", "sex", "age"], [["3", "female", "30"], ["1", "male", null]]));

        Assert.Equal(trained.Batch(Part.Train).FeatureNames, served.FeatureNames);
        Assert.Equal(2, served.RowCount);
        Assert.Equal(served.FeatureNames.Count, served.Features[0].Length);
        Assert.Equal([0, 1], served.HandedInAt);
    }

    [Fact]
    public void AServedRow_SaysWhichHandedInRowItIs_AfterTheWarmUpIsDropped()
    {
        IReadOnlyList<IReadOnlyList<string?>> Days(int count) =>
        [
            .. Enumerable.Range(0, count).Select(day => (IReadOnlyList<string?>)
            [
                new DateTime(2026, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                (100 + (day * 3 % 11)).ToString(CultureInfo.InvariantCulture),
            ]),
        ];

        var trained = Pdd.Create()
            .Read(new InMemoryRowSource(["Date", "close"], Days(60)), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("close"))
            .OrderBy("Date")
            .AddIndicator("sma5", Indicator.Sma, ["close"], 5)
            .DropWarmUp()
            .SplitByTime("Date", 0.70)
            .Normalise("close")
            .Drop("Date")
            .Build()
            .Run();

        var served = trained.Served(new InMemoryRowSource(["Date", "close"], Days(10)));

        Assert.Equal([4, 5, 6, 7, 8, 9], served.HandedInAt);
    }

    [Fact]
    public void AServedRowWithAGap_IsRefusedAsATrainingRowWouldBe()
    {
        var trained = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .Normalise("age")
            .Target("survived")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(
            () => trained.Served(new InMemoryRowSource(["age"], [["30"], [null]])));

        Assert.Contains("Row 2 of 'age' is still a gap", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AServedRowStillHoldingWords_IsRefused()
    {
        var trained = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Text("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Build()
            .Run();

        Assert.Throws<InvalidOperationException>(
            () => trained.Served(new InMemoryRowSource(["sex"], [["female"]])));
        Assert.Throws<ArgumentNullException>(() => Handover.Served(null!, new InMemoryRowSource(["sex"], [])));
        Assert.Throws<ArgumentNullException>(() => trained.Served(null!));
    }
}
