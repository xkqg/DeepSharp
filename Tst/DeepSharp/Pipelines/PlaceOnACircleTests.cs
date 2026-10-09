// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Where a value stands on a circle: a moment on the cycle of its hour, week, month, year or season, as 0.8.0 placed it
/// bit for bit where it placed it, and a number on a cycle of a length that is said and never worked out from the data.
/// </summary>
public class PlaceOnACircleTests
{
    // Moments and the place each cycle puts them at, with the length it counts them against. The first four cycles are 0.8.0's:
    // their places are held to the bit, since a model trained on them was trained on exactly those numbers.
    private static readonly string[] Written =
    [
        "2024-03-05T14:30:00",   // a Tuesday in a leap year, the 65th day
        "2023-03-05T00:00:00",   // the same date in a year that is not, the 64th day
        "2024-12-31T23:59:00",   // the last minute of a leap year
        "2024-01-01T00:00:00",   // the first
        "2025-12-01T06:00:00",   // winter begins in December
        "2025-06-15T12:00:00",   // summer
        "2025-09-01T00:00:00",   // autumn
    ];

    private static readonly Dictionary<Period, (double Place, double Length)[]> Places = new()
    {
        [Period.HourOfDay] = [(14.5, 24), (0, 24), (23 + (59 / 60.0), 24), (0, 24), (6, 24), (12, 24), (0, 24)],
        [Period.DayOfWeek] = [(2, 7), (0, 7), (2, 7), (1, 7), (1, 7), (0, 7), (1, 7)],
        [Period.DayOfMonth] = [(4, 31), (4, 31), (30, 31), (0, 31), (0, 31), (14, 31), (0, 31)],
        [Period.MonthOfYear] = [(2, 12), (2, 12), (11, 12), (0, 12), (11, 12), (5, 12), (8, 12)],
        [Period.DayOfYear] = [(64, 366), (63, 365), (365, 366), (0, 366), (334, 365), (165, 365), (243, 365)],
        [Period.Season] = [(1, 4), (1, 4), (0, 4), (0, 4), (0, 4), (2, 4), (3, 4)],
    };

    public static TheoryData<Period> Periods => [.. Enum.GetValues<Period>()];

    [Fact]
    public void EveryCycleThereIs_HasItsPlacesHere_SoACycleAddedLater_CannotBeLeftWithoutOne() =>
        Assert.Equal(Enum.GetValues<Period>().Order(), Places.Keys.Order());

    [Theory]
    [MemberData(nameof(Periods))]
    public void EveryMomentStandsWhereItsCycleSaysAndSinAndCosineAreWorkedOutFromThatPlaceAlone(Period period)
    {
        var table = Moments();

        new CyclicalStep("when", period).AddTo(table);

        var stem = $"when_{period.ToString().ToLowerInvariant()}";
        var sin = (Column<double>)table[$"{stem}_sin"];
        var cos = (Column<double>)table[$"{stem}_cos"];

        for (var row = 0; row < Written.Length; row++)
        {
            var (place, length) = Places[period][row];

            Assert.Equal(Math.Sin(2 * Math.PI * place / length), sin[row]);
            Assert.Equal(Math.Cos(2 * Math.PI * place / length), cos[row]);
        }
    }

    [Fact]
    public void ALeapYearIsOneTurn_SoTheLastDayIsNextToTheFirstAndNotOnIt()
    {
        var table = Moments();

        new CyclicalStep("when", Period.DayOfYear).AddTo(table);

        var sin = (Column<double>)table["when_dayofyear_sin"];
        var cos = (Column<double>)table["when_dayofyear_cos"];

        // 31 December of 2024 is 365 of 366 days along; 1 January of 2024 is at the start. One day apart, and not the same.
        var gap = Math.Sqrt(Math.Pow(sin[2]!.Value - sin[3]!.Value, 2) + Math.Pow(cos[2]!.Value - cos[3]!.Value, 2));

        Assert.InRange(gap, 2 * Math.Sin(Math.PI / 366) * 0.99, 2 * Math.Sin(Math.PI / 366) * 1.01);
    }

    [Theory]
    [InlineData("dayofyear", Period.DayOfYear)]
    [InlineData("season", Period.Season)]
    [InlineData("hourofday", Period.HourOfDay)]
    public void AFileSaysTheCycleByItsWord_AndItIsWrittenBackByIt(string word, Period period)
    {
        var json = $$"""{"step":"feature.cyclical","column":"when","period":"{{word}}","form":"signed"}""";
        var step = (CyclicalStep)StepCatalog.BuiltIn().ReadStep(json);

        Assert.Equal(period, step.Period);
        Assert.Equal(word, step.Period.Word());
    }

    [Fact]
    public void ACycleThatIsNoneOfThem_IsRefusedWhereItIsWritten() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new CyclicalStep("when", (Period)99));

    [Fact]
    public void AVerbThatExisted_KeepsTheVersionItMeansWhatItMeantFrom_AndTheNewVerbIsNewInTheEighth()
    {
        var catalog = StepCatalog.BuiltIn();

        Assert.Equal(1, catalog.Describe("feature.cyclical").Since);
        Assert.Equal(PipelineDeclaration.Version, catalog.Describe("feature.cycle").Since);
        Assert.Equal(8, catalog.Describe("feature.cycle").Since);
    }

    // ---- a number on a cycle of a length the person says ----

    [Fact]
    public void ANumberStandsWhereItFallsInTheLength_AndTheColumnsAreNamedByIt()
    {
        var table = Numbers("age\n0\n3\n7\n8\n-1\n7.5\n\n");

        new CycleStep("age", 7).AddTo(table);

        var sin = (Column<double>)table["age_cycle7_sin"];
        var cos = (Column<double>)table["age_cycle7_cos"];
        double[] places = [0, 3, 0, 1, 6, 0.5];

        for (var row = 0; row < places.Length; row++)
        {
            Assert.Equal(Math.Sin(2 * Math.PI * places[row] / 7), sin[row]!.Value, 12);
            Assert.Equal(Math.Cos(2 * Math.PI * places[row] / 7), cos[row]!.Value, 12);
        }

        Assert.True(table["age_cycle7_sin"].IsMissing(6), "A value that is missing stays missing: nothing invents a place.");
    }

    [Fact]
    public void ALengthWithAFraction_IsNamedByItsDigits_InvariantlyWhateverTheMachine()
    {
        var table = Numbers("day\n0\n1\n");

        new CycleStep("day", 29.53).AddTo(table);

        Assert.True(table.Has("day_cycle29.53_sin"));
        Assert.True(table.Has("day_cycle29.53_cos"));
    }

    [Theory]
    [InlineData(Form.Signed)]
    [InlineData(Form.Unit)]
    [InlineData(Form.SplitSign)]
    public void EveryFormWritesANumberAsItWritesAMoment(Form form)
    {
        var table = Numbers("age\n1\n5\n");

        new CycleStep("age", 7, form).AddTo(table);

        var name = form == Form.SplitSign ? "age_cycle7_sin_pos" : "age_cycle7_sin";

        Assert.True(table.Has(name));
        Assert.Equal(form, new CycleStep("age", 7, form).Form);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-7)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ALengthThatIsNotAboveNought_IsRefused_NotTakenFromTheData(double length) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new CycleStep("age", length));

    [Fact]
    public void ACycleNeedsAColumnToPlace() =>
        Assert.Throws<ArgumentException>(() => new CycleStep(" ", 7));

    [Fact]
    public void ACycleOfATextColumnOrAMomentIsAFaultOfTheDeclaration_NotOfTheRun()
    {
        var chain = Pdd.Create().ReadCsv("never-opened.csv").Declare(schema => schema.Timestamp("when").Category("farm").Number("age"));

        Assert.Throws<DeclarationException>(() => chain.Cycle("when", 7));
        Assert.Throws<DeclarationException>(() => chain.Cycle("farm", 7));
        Assert.Equal(["age"], chain.Cycle("age", 7).Declaration.Steps.OfType<CycleStep>().Select(step => step.Column));
    }

    [Fact]
    public void ACycleOfANumberIsWrittenAsAFileAndReadBackAsTheSameStep()
    {
        var declaration = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("age")).Cycle("age", 7, Form.Unit).Declaration;

        var json = declaration.ToJson();
        var again = PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn());

        Assert.Contains("\"step\": \"feature.cycle\"", json, StringComparison.Ordinal);
        Assert.Contains("\"length\": 7", json, StringComparison.Ordinal);
        Assert.Equal(declaration, again);
    }

    [Fact]
    public void OneCycleLine_PlacesEachColumnOnTheLengthItWasNamedUnder()
    {
        var steps = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("age", "day", "phase"))
            .Cycle(cycle => cycle.Every(7).Of("age", "day").Every(30).Of("phase"))
            .Declaration.Steps.OfType<CycleStep>().ToArray();

        Assert.Equal(["age", "day", "phase"], steps.Select(step => step.Column));
        Assert.Equal([7.0, 7.0, 30.0], steps.Select(step => step.Length));
    }

    [Fact]
    public void TheCycleLine_WritesThePlaceAsTheFeaturesLandWhereThePipelineSaidTheyDo()
    {
        var steps = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("age", "phase"))
            .DefaultFeatures(Form.Unit)
            .Cycle(cycle => cycle.Every(7).Of("age").Every(30).Of("phase"))
            .Declaration.Steps.OfType<CycleStep>().ToArray();

        Assert.Equal([Form.Unit, Form.Unit], steps.Select(step => step.Form));
    }

    [Fact]
    public void ACycleLineAfterItsColumnsAreNamed_NeedsTheColumns() =>
        Assert.Throws<ArgumentNullException>(() => Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("age")).Cycle(cycle => cycle.Every(7).Of(null!)));

    [Theory]
    [InlineData("""{"step":"feature.cycle","column":"age","form":"signed"}""")]
    [InlineData("""{"step":"feature.cycle","length":7,"form":"signed"}""")]
    [InlineData("""{"step":"feature.cycle","column":"age","length":0,"form":"signed"}""")]
    [InlineData("""{"step":"feature.cycle","column":"age","length":"seven","form":"signed"}""")]
    [InlineData("""{"step":"feature.cycle","column":"age","length":7,"form":"signed","period":"hourofday"}""")]
    public void AFileThatLeavesOutWhatACycleNeedsOrSaysWhatItDoesNotTake_IsRefused(string json) =>
        Assert.Throws<PipelineFileException>(() => StepCatalog.BuiltIn().ReadStep(json));

    [Fact]
    public void WhereTheFeaturesLand_GovernsACycleOfANumberAsItGovernsACycleOfAMoment()
    {
        var unit = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("age")).DefaultFeatures(Form.Unit).Cycle("age", 7);

        Assert.Equal([Form.Unit], unit.Declaration.Steps.OfType<CycleStep>().Select(step => step.Form));

        var said = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("age")).Cycle("age", 7);

        var refused = Assert.Throws<InvalidOperationException>(() => said.DefaultFeatures(Form.Unit));

        Assert.Contains("feature.cycle", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMomentAndANumberOnACircleAreBothSteps_ThatPlaceAColumnOnACircle()
    {
        IPlacesOnACircle[] both = [new CyclicalStep("when", Period.HourOfDay, Form.Unit), new CycleStep("age", 7, Form.SplitSign)];

        Assert.Equal([Form.Unit, Form.SplitSign], both.Select(step => step.Form));
    }

    private static Table Moments() => SchemaBinding.Bind(
        new DeclareStep([new ColumnDeclaration("when", ColumnKind.Timestamp, false)]),
        CsvRowSource.FromText("when\n" + string.Join('\n', Written) + "\n"));

    private static Table Numbers(string csv) => SchemaBinding.Bind(
        new DeclareStep([new ColumnDeclaration(csv.Split('\n')[0], ColumnKind.Number, true)]),
        CsvRowSource.FromText(csv));
}
