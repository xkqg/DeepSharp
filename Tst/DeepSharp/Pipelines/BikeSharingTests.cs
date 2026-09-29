// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The third published dataset: two years of a bike-sharing scheme, a row a day — its weather, its season and how the
/// day's rentals spread over its twenty-four hours. The spread is the answer, one answer of twenty-four shares, and the
/// day's total is what the shares come back as bikes by.
/// </summary>
public class BikeSharingTests
{
    private static readonly string[] Hours = [.. Enumerable.Range(0, 24).Select(hour => $"h{hour:00}")];

    private static PreparedData Days() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("bikes.csv"))
            .Declare(schema => schema
                .Timestamp("dteday")
                .Category("season", "weathersit")
                .Boolean("holiday", "workingday")
                .Number("temp", "atemp", "hum", "windspeed", "cnt")
                .Number(Hours))
            .SplitByTime("dteday", train: 0.70, validation: 0.15)
            .NormaliseRow(Norm.L1, Hours)
            .EncodeCategories()
            .Drop("dteday")
            .Distribution(Hours, scaleBy: "cnt")
            .Drop("cnt")
            .Build()
            .Run();

    [Fact]
    public void EveryDayOfTwoYears_IsARow_AndEachDaysShares_SumToOne()
    {
        var days = Days();
        var train = days.Batch(Part.Train);

        Assert.Equal(731, days.Table.RowCount);
        Assert.Equal(512, days.CountIn(Part.Train));
        Assert.Equal(110, days.CountIn(Part.Validation));
        Assert.Equal(109, days.CountIn(Part.Test));
        Assert.Equal(Hours, train.AnswerNames);
        Assert.All(train.Answers!, shares => Assert.Equal(1, shares.Sum(), 9));
    }

    [Fact]
    public void TheFirstDaysShares_ComeBackAsTheBikesRentedInEachOfItsHours()
    {
        var days = Days();
        var train = days.Batch(Part.Train);

        var bikes = days.BackToOriginal(train.Answers!, Part.Train)[0];

        Assert.Equal(
            [16, 40, 32, 13, 1, 1, 2, 3, 8, 14, 36, 56, 84, 94, 106, 110, 93, 67, 35, 37, 36, 34, 28, 39],
            bikes.Select(hour => Math.Round(hour, 9)));
        Assert.Equal(985, bikes.Sum(), 9);
    }
}
