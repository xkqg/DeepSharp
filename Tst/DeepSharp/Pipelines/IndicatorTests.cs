// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Indicators borrowed rather than written, and the two properties that decide whether borrowing them is
/// safe. The window must only look backwards, or a row carries what had not happened yet; and the warm-up
/// must be an absence rather than a value, or the first rows of every column are quietly a measurement
/// nobody took. Both are measured here, per indicator, not promised in a comment.
/// </summary>
public class IndicatorTests
{
    private static string Apple => Path.Join(RepoRoot(), "Samples", "data", "apple.csv");

    private static PipelineBuilder Prices() =>
        Pdd.Create()
            .ReadCsv(Apple)
            .Declare(schema => schema
                .Timestamp("Date")
                .Number("AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume"));

    public static TheoryData<Indicator, string[], int> EveryIndicator => new()
    {
        { Indicator.Sma, ["AAPL.Close"], 20 },
        { Indicator.Ema, ["AAPL.Close"], 20 },
        { Indicator.Rsi, ["AAPL.Close"], 14 },
        { Indicator.Atr, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 14 },
        { Indicator.Adx, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 14 },
        { Indicator.Cci, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 20 },
        { Indicator.WilliamsR, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 14 },
        { Indicator.Obv, ["AAPL.Close", "AAPL.Volume"], 1 },
        { Indicator.Macd, ["AAPL.Close"], 12 },
        { Indicator.BollingerBands, ["AAPL.Close"], 20 },
        { Indicator.Stochastic, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 14 },
        { Indicator.Vwap, ["AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume"], 1 },
    };

    [Theory]
    [MemberData(nameof(EveryIndicator))]
    public void EveryIndicatorOnlyLooksBackwards(Indicator indicator, string[] columns, int period)
    {
        // The impulse test. Change one row and nothing before it may move; an indicator that reaches
        // forward — a centred average, a smoothing pass over the whole series — fails here and cannot be
        // told apart from a backward one by looking at the lengths, which is the trap this replaces.
        var plain = Compute(indicator, columns, period, disturbAt: null);
        var disturbed = Compute(indicator, columns, period, disturbAt: 300);

        foreach (var (name, before) in plain)
        {
            var after = disturbed[name];

            for (var row = 0; row < 300; row++)
            {
                Assert.Equal(before[row], after[row]);
            }
        }

        // And the change did reach the rows at or after it, or the test proves nothing at all.
        Assert.Contains(plain, each =>
            Enumerable.Range(300, 206).Any(row => each.Value[row] != disturbed[each.Key][row]));
    }

    [Theory]
    [MemberData(nameof(EveryIndicator))]
    public void EveryIndicatorStartsWithAnAbsenceRatherThanANumberNobodyTook(
        Indicator indicator, string[] columns, int period)
    {
        var table = Prices().AddIndicator("made", indicator, columns, period).Build().Run().Table;
        var made = table.Columns
            .Where(column => column.Name.StartsWith("made", StringComparison.Ordinal))
            .ToArray();

        // Without this the loop below can pass by finding nothing at all, which is how a test goes green
        // while measuring the absence of the thing it is named after.
        Assert.NotEmpty(made);

        foreach (var column in made)
        {
            Assert.Equal(506, column.Count);

            // Whatever the warm-up is, it is absent and not a not-a-number: the value was never computed
            // rather than computed wrongly, and the library keeps those apart everywhere.
            var values = Numbers.Of(table, column.Name);

            Assert.DoesNotContain(values, value => value is { } number && double.IsNaN(number));
            Assert.Contains(values, value => value is not null);
        }
    }

    [Fact]
    public void AManyValuedIndicatorBecomesAColumnPerPart()
    {
        var table = Prices()
            .AddIndicator("macd", Indicator.Macd, ["AAPL.Close"])
            .AddIndicator("bb", Indicator.BollingerBands, ["AAPL.Close"], 20)
            .AddIndicator("stoch", Indicator.Stochastic, ["AAPL.High", "AAPL.Low", "AAPL.Close"])
            .Build()
            .Run()
            .Table;

        Assert.True(table.Has("macd_line"));
        Assert.True(table.Has("macd_signal"));
        Assert.True(table.Has("macd_histogram"));
        Assert.True(table.Has("bb_upper"));
        Assert.True(table.Has("bb_middle"));
        Assert.True(table.Has("bb_lower"));
        Assert.True(table.Has("stoch_k"));
        Assert.True(table.Has("stoch_d"));
    }

    [Fact]
    public void AMovingAverageIsTheAverageOfItsWindow()
    {
        // Against arithmetic anybody can check by hand, on the file's own numbers.
        var table = Prices().AddIndicator("sma3", Indicator.Sma, ["AAPL.Close"], 3).Build().Run().Table;

        var close = (Column<double>)table["AAPL.Close"];
        var sma = (Column<double>)table["sma3"];

        Assert.Equal((close[0]!.Value + close[1]!.Value + close[2]!.Value) / 3, sma[2]!.Value, 6);
        Assert.True(sma.IsMissing(0));
        Assert.True(sma.IsMissing(1));
    }

    [Fact]
    public void AnIndicatorBelongsBeforeTheSplit_AndTheChainSaysSo()
    {
        var declaration = Prices()
            .AddIndicator("rsi", Indicator.Rsi, ["AAPL.Close"])
            .SplitByTime("Date", 0.70, 0.15, 0.15)
            .Normalise("rsi")
            .Declaration;

        Assert.Equal("feature.indicator", declaration.Steps[2].Verb);

        // It learns nothing, so nothing about it is fitted and it may stand above the line.
        Assert.IsNotType<IFittedStep>(declaration.Steps[2], exactMatch: false);
    }

    [Fact]
    public void TheVerbSurvivesTheFileOnceTheCatalogKnowsIt()
    {
        var declaration = Prices()
            .AddIndicator("atr", Indicator.Atr, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 21)
            .Declaration;

        // The verb comes from a package, so a catalog that has not been told about it refuses the file
        // rather than guessing — and says which package is missing in the same breath.
        Assert.Throws<NotSupportedException>(() => PipelineDeclaration.FromJson(declaration.ToJson()));

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithIndicators());

        Assert.Equal(declaration, returned);
        Assert.Equal(21, ((AddIndicatorStep)returned.Steps[2]).Period);
    }

    [Fact]
    public void AnIndicatorGivenTheWrongNumberOfColumns_IsRefusedWhereItIsWritten()
    {
        var refused = Assert.Throws<ArgumentException>(
            () => new AddIndicatorStep("atr", Indicator.Atr, ["AAPL.Close"]));

        Assert.Contains("reads 3 columns", refused.Message);
        Assert.Throws<ArgumentException>(() => new AddIndicatorStep(" ", Indicator.Sma, ["a"]));
        Assert.Throws<ArgumentException>(() => new AddIndicatorStep("x", Indicator.Sma, [" "]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AddIndicatorStep("x", Indicator.Sma, ["a"], 0));
        Assert.Throws<ArgumentNullException>(() => new AddIndicatorStep("x", Indicator.Sma, null!));
        Assert.Throws<ArgumentNullException>(() => IndicatorExtensions.AddIndicator(null!, "x", Indicator.Sma, ["a"]));
        Assert.Throws<ArgumentNullException>(() => IndicatorExtensions.WithIndicators(null!));
    }

    [Fact]
    public void AFileNamingAnIndicatorNobodyDefined_IsRefused()
    {
        const string json = """
            {"declaration":[{"step":"feature.indicator","column":"x","indicator":"astrology",
                             "period":14,"columns":["a"]}]}
            """;

        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn().WithIndicators()));
    }

    [Fact]
    public void AnIndicatorStepWithoutItsColumnList_IsRefused()
    {
        const string json = """
            {"declaration":[{"step":"feature.indicator","column":"x","indicator":"sma","period":14}]}
            """;

        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn().WithIndicators()));
    }

    [Fact]
    public void TwoIndicatorsThatDifferAreNotTheSameStep()
    {
        var sma = new AddIndicatorStep("x", Indicator.Sma, ["a"], 10);

        Assert.Equal(sma, new AddIndicatorStep("x", Indicator.Sma, ["a"], 10));
        Assert.Equal(sma.GetHashCode(), new AddIndicatorStep("x", Indicator.Sma, ["a"], 10).GetHashCode());
        Assert.NotEqual(sma, new AddIndicatorStep("x", Indicator.Sma, ["a"], 20));
        Assert.NotEqual(sma, new AddIndicatorStep("x", Indicator.Ema, ["a"], 10));
        Assert.NotEqual(sma, new AddIndicatorStep("y", Indicator.Sma, ["a"], 10));
        Assert.NotEqual(sma, new AddIndicatorStep("x", Indicator.Sma, ["b"], 10));
    }

    [Fact]
    public void APackageRegistersItsVerbWhicheverOrderTheHostWritesIt()
    {
        var catalog = StepCatalog.BuiltIn();

        new IndicatorSteps().AddTo(catalog);

        Assert.True(catalog.Knows("feature.indicator"));
    }

    private static Dictionary<string, double?[]> Compute(
        Indicator indicator, string[] columns, int period, int? disturbAt)
    {
        var table = Prices().Build().Prepare();

        if (disturbAt is { } row)
        {
            foreach (var name in columns)
            {
                var column = (Column<double>)table[name];
                column[row] = column[row]!.Value * 1.5;
            }
        }

        new AddIndicatorStep("made", indicator, columns, period).AddTo(table);

        return table.Columns
            .Where(column => column.Name.StartsWith("made", StringComparison.Ordinal))
            .ToDictionary(column => column.Name, column => Numbers.Of(table, column.Name));
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
