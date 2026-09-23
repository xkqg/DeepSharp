// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A month is not a quantity. March is not three of anything and December is not twelve times January, so
/// the pieces of a moment in time arrive as categories and the encoder takes them from there. The two
/// exceptions are said out loud: a piece where the order really is the point, and a piece that wraps round,
/// which a circle says better than a category ever can.
/// </summary>
public class TimePartTests
{
    private static Table Moments(params string[] written)
    {
        var source = CsvRowSource.FromText($"when\n{string.Join("\n", written)}\n");

        return SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("when", ColumnKind.Timestamp, true)]), source);
    }

    [Fact]
    public void EveryPieceOfAMomentIsTakenOutUnderItsOwnName()
    {
        var table = Moments("2026-03-15T14:37:00");

        new TimePartsStep("when", [
            TimePart.Minute, TimePart.Hour, TimePart.DayOfWeek, TimePart.DayOfMonth,
            TimePart.Month, TimePart.Quarter, TimePart.Season, TimePart.Year]).AddTo(table);

        Assert.Equal("37", table["when_minute"].TextAt(0));
        Assert.Equal("14", table["when_hour"].TextAt(0));
        Assert.Equal("sunday", table["when_dayofweek"].TextAt(0));
        Assert.Equal("15", table["when_dayofmonth"].TextAt(0));
        Assert.Equal("3", table["when_month"].TextAt(0));
        Assert.Equal("1", table["when_quarter"].TextAt(0));
        Assert.Equal("spring", table["when_season"].TextAt(0));
        Assert.Equal("2026", table["when_year"].TextAt(0));
    }

    [Fact]
    public void AndEveryOneOfThemIsACategoryUnlessYouSayOtherwise()
    {
        var table = Moments("2026-03-15T14:37:00");
        var step = new TimePartsStep("when", [TimePart.Month, TimePart.Season]);

        step.AddTo(table);

        Assert.Equal(ColumnKind.Category, table["when_month"].Kind);
        Assert.Equal(ColumnKind.Category, table["when_season"].Kind);
        Assert.Equal(["when_month", "when_season"], step.Categories);
    }

    [Fact]
    public void AskedForAsNumbers_TheyAreNumbersAndNotCategories()
    {
        var table = Moments("2026-03-15T14:37:00");
        var step = new TimePartsStep("when", [TimePart.Year, TimePart.Month], asCategories: false);

        step.AddTo(table);

        Assert.Equal(ColumnKind.Number, table["when_year"].Kind);
        Assert.Equal(2026, ((Column<double>)table["when_year"])[0]);
        Assert.Empty(step.Categories);
    }

    [Theory]
    [InlineData("2026-01-15", "winter")]
    [InlineData("2026-02-15", "winter")]
    [InlineData("2026-12-15", "winter")]
    [InlineData("2026-04-15", "spring")]
    [InlineData("2026-07-15", "summer")]
    [InlineData("2026-10-15", "autumn")]
    public void TheSeasonsAreTheMeteorologicalOnes(string when, string season)
    {
        // A convention rather than a fact, and the northern one: December through February is winter.
        var table = Moments(when);

        new TimePartsStep("when", [TimePart.Season]).AddTo(table);

        Assert.Equal(season, table["when_season"].TextAt(0));
    }

    [Theory]
    [InlineData("2026-02-15", "1")]
    [InlineData("2026-05-15", "2")]
    [InlineData("2026-08-15", "3")]
    [InlineData("2026-11-15", "4")]
    public void TheQuartersRunThreeMonthsEach(string when, string quarter)
    {
        var table = Moments(when);

        new TimePartsStep("when", [TimePart.Quarter]).AddTo(table);

        Assert.Equal(quarter, table["when_quarter"].TextAt(0));
    }

    [Fact]
    public void ADayOfTheWeekIsWrittenAsItsName()
    {
        // A number would invite somebody to average it, and the average of Tuesday and Thursday is not
        // Wednesday — it is a category, and it is written as one.
        var table = Moments("2026-09-23");

        new TimePartsStep("when", [TimePart.DayOfWeek]).AddTo(table);

        Assert.Equal("wednesday", table["when_dayofweek"].TextAt(0));
    }

    [Fact]
    public void AMomentThatIsNotThere_LeavesEveryPieceOfItAbsent()
    {
        var table = Moments("2026-03-15", string.Empty);

        new TimePartsStep("when", [TimePart.Month, TimePart.Year], asCategories: false).AddTo(table);

        Assert.True(table["when_month"].IsMissing(1));
        Assert.True(table["when_year"].IsMissing(1));
        Assert.False(table["when_month"].IsMissing(0));
    }

    [Fact]
    public void TheEncoderTakesThemFromTheStepJustAsItTakesTheSchemasOwn()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Path.Join(RepoRoot(), "Samples", "data", "apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close").Category("direction"))
            .TimeParts("Date", TimePart.Season, TimePart.DayOfWeek)
            .SplitByTime("Date", 0.70, 0.15)
            .EncodeCategories()
            .Build()
            .Run();

        // One word in the schema, one verb for the moment, and the encoder found all three without being
        // told a second time.
        Assert.True(prepared.Table.Has("direction_Increasing"));
        Assert.True(prepared.Table.Has("Date_season_winter"));
        Assert.True(prepared.Table.Has("Date_dayofweek_monday"));
        Assert.False(prepared.Table.Has("Date_season"));
    }

    [Fact]
    public void TakingApartSomethingThatIsNotAMoment_IsRefused()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]),
            CsvRowSource.FromText("a\n1\n"));

        var refused = Assert.Throws<InvalidOperationException>(
            () => new TimePartsStep("a", [TimePart.Year]).AddTo(table));

        Assert.Contains("no moment in time", refused.Message);
    }

    [Fact]
    public void AStepWithoutAColumnOrWithoutPieces_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new TimePartsStep(" ", [TimePart.Year]));
        Assert.Throws<ArgumentException>(() => new TimePartsStep("when", []));
        Assert.Throws<ArgumentNullException>(() => new TimePartsStep("when", null!));
    }

    [Fact]
    public void TheVerbSurvivesTheFile()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Timestamp("when"))
            .TimeParts("when", TimePart.Hour, TimePart.Season)
            .TimePartsAsNumbers("when", TimePart.Year)
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson());

        Assert.Equal(declaration, returned);
        Assert.True(((TimePartsStep)returned.Steps[2]).AsCategories);
        Assert.False(((TimePartsStep)returned.Steps[3]).AsCategories);
    }

    [Theory]
    [InlineData("""{"declaration":[{"step":"feature.timeParts","column":"w","asCategories":true,"parts":["fortnight"]}]}""")]
    [InlineData("""{"declaration":[{"step":"feature.timeParts","column":"w","asCategories":true}]}""")]
    public void AFileNamingAPieceNobodyDefined_IsRefused(string json)
    {
        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));
    }

    [Fact]
    public void TwoStepsThatDifferAreNotTheSameStep()
    {
        var one = new TimePartsStep("when", [TimePart.Year]);

        Assert.Equal(one, new TimePartsStep("when", [TimePart.Year]));
        Assert.Equal(one.GetHashCode(), new TimePartsStep("when", [TimePart.Year]).GetHashCode());
        Assert.NotEqual(one, new TimePartsStep("when", [TimePart.Month]));
        Assert.NotEqual(one, new TimePartsStep("other", [TimePart.Year]));
        Assert.NotEqual(one, new TimePartsStep("when", [TimePart.Year], asCategories: false));
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "DeepSharp.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return directory!.FullName;
    }
}
