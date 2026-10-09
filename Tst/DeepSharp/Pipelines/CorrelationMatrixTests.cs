// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The correlation the pipeline works out once, from the rows it kept: Pearson's, and Spearman's on average ranks, as a
/// number for every pair of columns and an honest NaN where no coefficient is defined. Every reference value is what numpy
/// and scipy answered, printed by Fixtures/correlation-reference.py into correlation-reference.txt beside it.
/// </summary>
public class CorrelationMatrixTests
{
    // numpy 2.5.3 and scipy 1.18.1: five columns over ten rows with ties — x, y, exp(x), -x and x again.
    private static readonly double[] X = [1, 2, 2, 3, 5, 5, 5, 9, 10, 12];
    private static readonly double[] Y = [2, 1, 2, 4, 4, 6, 7, 8, 7, 11];

    private static readonly double[][] FlockPearson =
    [
        [0.9999999999999999, 0.922749770775667, 0.7017873557233801, -0.9999999999999999, 0.9999999999999999],
        [0.922749770775667, 1.0, 0.6965380454794022, -0.922749770775667, 0.922749770775667],
        [0.7017873557233801, 0.6965380454794021, 1.0, -0.7017873557233801, 0.7017873557233801],
        [-0.9999999999999999, -0.922749770775667, -0.7017873557233801, 0.9999999999999999, -0.9999999999999999],
        [0.9999999999999999, 0.922749770775667, 0.7017873557233801, -0.9999999999999999, 0.9999999999999999],
    ];

    private static readonly double[][] FlockSpearman =
    [
        [1.0, 0.9223780407186633, 1.0, -1.0, 1.0],
        [0.9223780407186633, 1.0, 0.9223780407186633, -0.9223780407186633, 0.9223780407186633],
        [1.0, 0.9223780407186633, 1.0, -1.0, 1.0],
        [-1.0, -0.9223780407186633, -1.0, 1.0, -1.0],
        [1.0, 0.9223780407186633, 1.0, -1.0, 1.0],
    ];

    private static readonly string[] FlockNames = ["x", "y", "exp", "minus", "again"];

    private static CorrelationInput Flock() =>
        Input(FlockNames, X, Y, [.. X.Select(Math.Exp)], [.. X.Select(value => -value)], X);

    [Fact]
    public void Pearson_IsWhatNumpyAnswers_ForEveryPair()
    {
        var matrix = Flock().Correlate();

        Assert.Equal(FlockNames, matrix.Columns);
        Assert.Equal(10, matrix.Kept);
        Assert.Equal(5, matrix.Pearson.Size);
        AssertMatches(FlockPearson, matrix.Pearson);
    }

    [Fact]
    public void Spearman_IsWhatScipyAnswers_WithTiesRankedAsTheirAverage()
    {
        var matrix = Flock().Correlate();

        AssertMatches(FlockSpearman, matrix.Spearman);
    }

    [Fact]
    public void ACurveThatKeepsTheOrder_IsPerfectForRanks_AndNotForPearson()
    {
        var matrix = Flock().Correlate();

        Assert.Equal(1, matrix.Spearman["x", "exp"], 12);
        Assert.Equal(0.7017873557233801, matrix.Pearson["x", "exp"], 12);
    }

    [Fact]
    public void ColumnsThatAreTheSame_AreOne_AndOneThatRunsTheOtherWay_IsMinusOne()
    {
        var matrix = Flock().Correlate();

        Assert.Equal(1, matrix.Pearson["x", "again"], 12);
        Assert.Equal(1, matrix.Spearman["x", "again"], 12);
        Assert.Equal(-1, matrix.Pearson["x", "minus"], 12);
        Assert.Equal(-1, matrix.Spearman["x", "minus"], 12);
    }

    [Fact]
    public void TheDiagonalOfADefinedColumn_IsExactlyOne_AndNoCoefficientEverLeavesTheirScale()
    {
        var matrix = Flock().Correlate();

        Assert.All(Enumerable.Range(0, 5), at => Assert.Equal(1.0, matrix.Pearson[at, at]));
        Assert.All(Enumerable.Range(0, 5), at => Assert.Equal(1.0, matrix.Spearman[at, at]));
        Assert.All(matrix.Pearson.ToArray().SelectMany(row => row), value => Assert.InRange(value, -1.0, 1.0));
        Assert.All(matrix.Spearman.ToArray().SelectMany(row => row), value => Assert.InRange(value, -1.0, 1.0));
    }

    [Fact]
    public void WithTies_SpearmanIsPearsonOfTheAverageRanks_AndNotAnyOtherRankingOfThem()
    {
        // The eight rows scipy.stats.spearmanr is told by: min ranks would give 0.9142, max ranks 0.8427, dense ranks 0.8678.
        double[] first = [1, 2, 2, 3, 5, 5, 5, 9];
        double[] second = [2, 1, 2, 4, 4, 6, 7, 8];
        var matrix = Input(["x", "y"], first, second).Correlate();

        Assert.Equal(0.9007775105401477, matrix.Spearman["x", "y"], 12);
        Assert.Equal(0.8961882438253471, matrix.Pearson["x", "y"], 12);
    }

    [Fact]
    public void RanksDoNotMindACurve_WhereAPearsonDoes()
    {
        double[] first = [1, 2, 2, 3, 5, 5, 5, 9];
        double[] second = [2, 1, 2, 4, 4, 6, 7, 8];
        var matrix = Input(["curve", "y"], [.. first.Select(Math.Exp)], second).Correlate();

        Assert.Equal(0.9007775105401477, matrix.Spearman["curve", "y"], 12);
        Assert.Equal(0.610665316623519, matrix.Pearson["curve", "y"], 12);
    }

    [Fact]
    public void AColumnThatNeverChanges_HasNoCoefficient_NotNought()
    {
        var matrix = Input(["x", "constant", "y"], X, [.. X.Select(_ => 1.0)], Y).Correlate();

        foreach (var coefficients in new[] { matrix.Pearson, matrix.Spearman })
        {
            Assert.All(Enumerable.Range(0, 3), at => Assert.True(double.IsNaN(coefficients[1, at])));
            Assert.All(Enumerable.Range(0, 3), at => Assert.True(double.IsNaN(coefficients[at, 1])));
            Assert.Equal(1.0, coefficients[0, 0]);
            Assert.Equal(1.0, coefficients[2, 2]);
        }

        Assert.Equal(0.922749770775667, matrix.Pearson["x", "y"], 12);
        Assert.Equal(0.9223780407186633, matrix.Spearman["x", "y"], 12);
    }

    [Fact]
    public void AConstantColumn_IsConstantWhateverItsMeanDoesInFloatingPoint()
    {
        // Ten additions of 0.1, over ten, are not 0.1, so a mean worked out by adding leaves every value a hair from it:
        // the column must be found constant by what it holds, and not by a spread that rounding left over.
        var matrix = Input(["x", "tenth", "y"], X, [.. X.Select(_ => 0.1)], Y).Correlate();

        Assert.True(double.IsNaN(matrix.Pearson["x", "tenth"]));
        Assert.True(double.IsNaN(matrix.Pearson["tenth", "tenth"]));
        Assert.True(double.IsNaN(matrix.Spearman["tenth", "y"]));
        Assert.Equal(0.922749770775667, matrix.Pearson["x", "y"], 12);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void FewerThanTwoRows_CorrelateNothing_EveryCoefficientIsNaN(int rows)
    {
        var matrix = Input(["a", "b"], [.. X.Take(rows)], [.. Y.Take(rows)]).Correlate();

        Assert.Equal(rows, matrix.Kept);
        Assert.Equal(2, matrix.Pearson.Size);
        Assert.All(matrix.Pearson.ToArray().SelectMany(row => row), value => Assert.True(double.IsNaN(value)));
        Assert.All(matrix.Spearman.ToArray().SelectMany(row => row), value => Assert.True(double.IsNaN(value)));
    }

    [Fact]
    public void ARowWithAGap_IsLeftOutOfEveryPair_AndTheRowsKeptAreCounted()
    {
        // The rule is the input's own, complete rows, so the matrix is what the same data without that row gives.
        var withGap = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b", "c"], [["1", "2", "9"], ["2", "1", "7"], ["3", "", "8"], ["4", "4", "1"], ["5", "3", "2"]]), "five rows")
            .Declare(schema => schema.Number("a", "c").Optional("b", ColumnKind.Number))
            .Correlation(["a", "b", "c"])
            .Build()
            .Run()
            .Evidence.Values.OfType<CorrelationInput>().Single();
        var without = Input(["a", "b", "c"], [1, 2, 4, 5], [2, 1, 4, 3], [9, 7, 1, 2]);

        var matrix = withGap.Correlate();

        Assert.Equal(4, matrix.Kept);
        AssertMatches(without.Correlate().Pearson.ToArray(), matrix.Pearson);
        AssertMatches(without.Correlate().Spearman.ToArray(), matrix.Spearman);
    }

    [Fact]
    public void ACoefficient_IsAskedForByPlaceOrByName_AndBothSayTheSame()
    {
        var pearson = Flock().Correlate().Pearson;

        Assert.Equal(pearson[1, 2], pearson["y", "exp"]);
        Assert.Equal(pearson[2, 1], pearson["exp", "y"]);
        Assert.Equal(FlockNames, pearson.Columns);
    }

    [Fact]
    public void AnUnknownColumnOrPlace_IsRefused_NamingWhatIsThere()
    {
        var pearson = Flock().Correlate().Pearson;

        var unknown = Assert.Throws<ArgumentException>(() => pearson["x", "weight"]);
        Assert.Contains("weight", unknown.Message, StringComparison.Ordinal);
        Assert.Contains("exp", unknown.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => pearson["weight", "x"]);
        Assert.Throws<ArgumentOutOfRangeException>(() => pearson[5, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => pearson[0, -1]);
    }

    [Fact]
    public void TheRowsHandedOut_AreACopy_SoNobodyChangesTheNumbersOthersRead()
    {
        var pearson = Flock().Correlate().Pearson;
        var rows = pearson.ToArray();

        Assert.Equal(5, rows.Length);
        Assert.All(rows, row => Assert.Equal(5, row.Length));

        rows[0][1] = 99;

        Assert.Equal(0.922749770775667, pearson[0, 1], 12);
    }

    [Fact]
    public void EitherCoefficient_IsAskedForByItsWord()
    {
        var matrix = Flock().Correlate();

        Assert.Same(matrix.Pearson, matrix.Of(Coefficient.Pearson));
        Assert.Same(matrix.Spearman, matrix.Of(Coefficient.Spearman));
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.Of((Coefficient)7));
    }

    [Fact]
    public void TheMatrixIsWorkedOutOnce_NoMatterHowManyAskForIt()
    {
        var input = Flock();

        Assert.Same(input.Correlate(), input.Correlate());
    }

    [Fact]
    public void TheCoefficientsKnowNoLibraryOutsideThePipeline_AndNoClock()
    {
        // Pipelines brings nothing but its own: the numbers are worked out here, so a chart and a profile read the same ones.
        var references = typeof(CorrelationMatrix).Assembly.GetReferencedAssemblies().Select(reference => reference.Name);

        Assert.DoesNotContain(references, name => name!.StartsWith("MatPlotLibNet", StringComparison.Ordinal));
        Assert.DoesNotContain(references, name => name!.StartsWith("MathNet", StringComparison.Ordinal));
    }

    private static void AssertMatches(double[][] expected, CorrelationCoefficients actual)
    {
        Assert.Equal(expected.Length, actual.Size);

        for (var row = 0; row < expected.Length; row++)
        {
            for (var column = 0; column < expected.Length; column++)
            {
                Assert.True(
                    Math.Abs(expected[row][column] - actual[row, column]) < 1e-12,
                    string.Create(CultureInfo.InvariantCulture, $"[{row}, {column}] is {actual[row, column]:R}, expected {expected[row][column]:R}"));
            }
        }
    }

    // The rows a correlation is drawn from, as a run keeps them: one array of numbers per column, written as a file writes them.
    private static CorrelationInput Input(string[] names, params double[][] columns)
    {
        var count = columns[0].Length;
        string[][] rows =
        [
            .. Enumerable.Range(0, count).Select(row => columns.Select(column => column[row].ToString("R", CultureInfo.InvariantCulture)).ToArray()),
        ];
        var declared = Pdd.Create().Read(new InMemoryRowSource(names, rows), "rows")
            .Declare(schema => names.Aggregate(schema, (each, name) => each.Optional(name, ColumnKind.Number)))
            .Correlation(names);

        return declared.Build().Run().Evidence.Values.OfType<CorrelationInput>().Single();
    }
}
