// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Features, forms, categories and scales, on the two published files. Everything here either learns from
/// the training rows alone or learns nothing at all, and which of the two it is decides where in the chain
/// it is allowed to stand.
/// </summary>
public class TransformTests
{
    private static string Data(string file) => Path.Join(RepoRoot(), "Samples", "data", file);

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

    private static PreparedData Apple(Action<PipelineBuilder> before, Action<FittingBuilder>? after = null)
    {
        var builder = Pdd.Create()
            .ReadCsv(Data("apple.csv"))
            .Declare(schema => schema
                .Timestamp("Date")
                .Number("AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume")
                .Text("direction"));

        before(builder);

        var fitting = builder.SplitByTime("Date", 0.70, 0.15);
        after?.Invoke(fitting);

        return fitting.Build().Run();
    }

    [Fact]
    public void AFeature_IsWorkedOutFromTheColumnsItNames()
    {
        var prepared = Apple(builder => builder.AddFeature("range", "AAPL.High", Arithmetic.Minus, "AAPL.Low"));
        var range = (Column<double>)prepared.Table["range"];
        var high = (Column<double>)prepared.Table["AAPL.High"];
        var low = (Column<double>)prepared.Table["AAPL.Low"];

        Assert.Equal(high[0]!.Value - low[0]!.Value, range[0]!.Value, 10);
        Assert.Equal(506, range.Count);
    }

    [Theory]
    [InlineData(Arithmetic.Plus, 3)]
    [InlineData(Arithmetic.Minus, -1)]
    [InlineData(Arithmetic.Times, 2)]
    [InlineData(Arithmetic.DividedBy, 0.5)]
    public void EveryArithmeticDoesWhatItSays(Arithmetic arithmetic, double expected)
    {
        var table = Read("a,b\n1,2\n");

        new AddFeatureStep("c", "a", arithmetic, "b").AddTo(table);

        Assert.Equal(expected, ((Column<double>)table["c"])[0]);
    }

    [Fact]
    public void AGapOnEitherSide_LeavesAGapRatherThanANumber()
    {
        var table = Read("a,b\n1,\n,2\n3,4\n");
        var feature = new AddFeatureStep("c", "a", Arithmetic.Plus, "b");

        feature.AddTo(table);

        var made = (Column<double>)table["c"];

        Assert.True(made.IsMissing(0));
        Assert.True(made.IsMissing(1));
        Assert.Equal(7, made[2]);
    }

    [Fact]
    public void ADivisionByNothing_IsAGapAndNotAnInfinity()
    {
        var table = Read("a,b\n1,0\n");
        new AddFeatureStep("c", "a", Arithmetic.DividedBy, "b").AddTo(table);

        Assert.True(table["c"].IsMissing(0));
    }

    [Fact]
    public void AMomentInTime_BecomesAPlaceOnACircle()
    {
        var prepared = Apple(builder => builder.Cyclical("Date", Period.MonthOfYear));
        var sin = (Column<double>)prepared.Table["Date_monthofyear_sin"];
        var cos = (Column<double>)prepared.Table["Date_monthofyear_cos"];

        // The first row is February 2015: the second of twelve months, one twelfth of a turn along.
        Assert.Equal(Math.Sin(2 * Math.PI / 12), sin[0]!.Value, 10);
        Assert.Equal(Math.Cos(2 * Math.PI / 12), cos[0]!.Value, 10);
        Assert.Equal(506, sin.Count);
    }

    [Fact]
    public void DecemberAndJanuaryAreNeighbours_WhichIsTheWholePoint()
    {
        var table = Read("when\n2026-12-15\n2027-01-15\n2026-06-15\n", ColumnKind.Timestamp);
        new CyclicalStep("when", Period.MonthOfYear).AddTo(table);

        var sin = (Column<double>)table["when_monthofyear_sin"];
        var cos = (Column<double>)table["when_monthofyear_cos"];

        double Distance(int one, int other) =>
            Math.Sqrt(Math.Pow(sin[one]!.Value - sin[other]!.Value, 2)
                      + Math.Pow(cos[one]!.Value - cos[other]!.Value, 2));

        // December to January is one step; December to June is half the circle away.
        Assert.True(Distance(0, 1) < Distance(0, 2), "December should be nearer January than June");
    }

    [Fact]
    public void TheUnitForm_PutsEverythingBetweenNothingAndOne()
    {
        var table = Read("when\n2026-01-15\n2026-07-15\n", ColumnKind.Timestamp);
        new CyclicalStep("when", Period.MonthOfYear, Form.Unit).AddTo(table);

        var sin = (Column<double>)table["when_monthofyear_sin"];

        Assert.InRange(sin[0]!.Value, 0, 1);
        Assert.InRange(sin[1]!.Value, 0, 1);
    }

    [Fact]
    public void TheSplitSignForm_MakesTwoColumnsThatAddBackUp()
    {
        var table = Read("when\n2026-01-15\n2026-07-15\n2026-10-15\n", ColumnKind.Timestamp);
        new CyclicalStep("when", Period.MonthOfYear, Form.SplitSign).AddTo(table);

        var up = (Column<double>)table["when_monthofyear_sin_pos"];
        var down = (Column<double>)table["when_monthofyear_sin_neg"];

        for (var row = 0; row < table.RowCount; row++)
        {
            Assert.InRange(up[row]!.Value, 0, 1);
            Assert.InRange(down[row]!.Value, 0, 1);
            Assert.True(up[row] == 0 || down[row] == 0, "one half of a split sign is always nothing");
        }

        // Exactly reversible: what went up minus what went down is the value it came from.
        Assert.Equal(Math.Sin(2 * Math.PI * 9 / 12), up[2]!.Value - down[2]!.Value, 10);
    }

    [Theory]
    [InlineData(Scale.Standard)]
    [InlineData(Scale.MinMax)]
    [InlineData(Scale.MaxAbs)]
    [InlineData(Scale.Robust)]
    public void EveryScaleLearnsFromTheTrainingRowsAlone(Scale scale)
    {
        var prepared = Apple(_ => { }, fitting => fitting.Normalise("AAPL.Close", scale));
        var learned = prepared.Fitted.Values.Single();

        var closes = Enumerable.Range(0, prepared.Table.RowCount)
            .Where(row => prepared.Parts[row] == Part.Train)
            .Select(row => ((Column<double>)prepared.Table["AAPL.Close"])[row]!.Value)
            .ToArray();

        Assert.True(learned.Numbers.ContainsKey("centre"));
        Assert.True(learned.Numbers.ContainsKey("spread"));

        // Whatever the kind, the training rows come out centred near nothing and the scale is not the
        // scale of the test rows, which the fit never saw.
        Assert.InRange(closes.Average(), -1.5, 1.5);
    }

    [Fact]
    public void MinMaxOnAPriceMeetsANewHigh_AndSaysWhatItDoesAboutIt()
    {
        // The training rows are the early ones, so a later price is above everything the fit ever saw.
        var passed = Apple(_ => { }, fitting => fitting.Normalise("AAPL.Close", Scale.MinMax));
        var clipped = Apple(_ => { }, fitting => fitting.Normalise("AAPL.Close", Scale.MinMax, OutOfRange.Clip));

        var passedClose = (Column<double>)passed.Table["AAPL.Close"];
        var clippedClose = (Column<double>)clipped.Table["AAPL.Close"];

        var highest = Enumerable.Range(0, passed.Table.RowCount).Max(row => passedClose[row]!.Value);

        Assert.True(highest > 1, $"a later price should leave the learned range, and the highest was {highest}");
        Assert.Equal(1, Enumerable.Range(0, clipped.Table.RowCount).Max(row => clippedClose[row]!.Value));

        Assert.Throws<InvalidOperationException>(
            () => Apple(_ => { }, fitting => fitting.Normalise("AAPL.Close", Scale.MinMax, OutOfRange.Refuse)));
    }

    [Fact]
    public void ACategory_BecomesOneColumnPerCategoryPlusAPlaceForTheUnfamiliar()
    {
        var prepared = Apple(_ => { }, fitting => fitting.Encode("direction"));

        Assert.False(prepared.Table.Has("direction"));
        Assert.True(prepared.Table.Has("direction_Increasing"));
        Assert.True(prepared.Table.Has("direction_Decreasing"));
        Assert.True(prepared.Table.Has("direction_other"));

        var increasing = (Column<double>)prepared.Table["direction_Increasing"];
        var decreasing = (Column<double>)prepared.Table["direction_Decreasing"];

        Assert.Equal(1, increasing[0]);
        Assert.Equal(0, decreasing[0]);
        Assert.Equal(["Decreasing", "Increasing"], prepared.Fitted.Values.Single().List("categories"));
    }

    [Fact]
    public void AnOrdinalEncoding_IsOneColumnOfPlaces()
    {
        var prepared = Apple(_ => { }, fitting => fitting.Encode("direction", As.Ordinal));

        Assert.Equal(ColumnKind.Number, prepared.Table["direction"].Kind);
        Assert.Equal(1, ((Column<double>)prepared.Table["direction"])[0]);
    }

    [Fact]
    public void ACategoryTheTrainingRowsNeverHeld_IsRefusedWhenYouSaidToRefuseIt()
    {
        var table = Read("kind\na\nb\nz\n", ColumnKind.Text);
        var step = new EncodeStep("kind", As.Ordinal, Unseen.Refuse);
        var parts = new[] { Part.Train, Part.Train, Part.Test };

        var learned = step.Fit(table, parts);

        var refused = Assert.Throws<InvalidOperationException>(() => step.ApplyTo(table, learned));

        Assert.Contains("'z'", refused.Message);
    }

    [Fact]
    public void ScalingARow_LeavesItsDirectionAndTakesItsSize()
    {
        var table = Read("a,b\n3,4\n6,8\n");
        new NormaliseRowStep(["a", "b"]).AddTo(table);

        var a = (Column<double>)table["a"];
        var b = (Column<double>)table["b"];

        // Three and four make a row of length five; six and eight make one of ten. Both point the same way.
        Assert.Equal(0.6, a[0]!.Value, 10);
        Assert.Equal(0.8, b[0]!.Value, 10);
        Assert.Equal(0.6, a[1]!.Value, 10);
        Assert.Equal(0.8, b[1]!.Value, 10);
    }

    [Theory]
    [InlineData(Norm.L1, 1.0 / 3.0)]
    [InlineData(Norm.Max, 0.5)]
    public void EveryNormMeasuresTheRowItsOwnWay(Norm norm, double expected)
    {
        var table = Read("a,b\n1,2\n");
        new NormaliseRowStep(["a", "b"], norm).AddTo(table);

        Assert.Equal(expected, ((Column<double>)table["a"])[0]!.Value, 10);
    }

    [Fact]
    public void EveryNewVerbSurvivesTheFile()
    {
        var declaration = Pdd.Create()
            .ReadCsv("apple.csv")
            .Declare(schema => schema.Timestamp("Date").Number("high", "low").Text("direction"))
            .AddFeature("range", "high", Arithmetic.Minus, "low")
            .Cyclical("Date", Period.DayOfWeek, Form.SplitSign)
            .SplitByTime("Date", 0.70, 0.15)
            .FillMissing("range", With.Median)
            .Normalise("high", Scale.Robust, OutOfRange.Clip)
            .Encode("direction", As.Ordinal, Unseen.Refuse)
            .NormaliseRow(Norm.L1, "high", "low")
            .Declaration;

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson()));
        Assert.Equal(9, declaration.Steps.Count);
    }

    private static Table Read(string csv, ColumnKind kind = ColumnKind.Number)
    {
        var source = CsvRowSource.FromText(csv);
        var schema = new SchemaBuilder();

        foreach (var name in source.ColumnNames)
        {
            schema.Column(name, kind, optional: true);
        }

        return SchemaBinding.Bind(new DeclareStep(schema.Columns), source);
    }

    // ---- the limits of the bell-curve shape ----------------------------------------------------------

    private static Table OneColumn(params double[] values) => SchemaBinding.Bind(
        new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
        CsvRowSource.FromText(
            "a\n" + string.Join("\n", values.Select(
                value => value.ToString("R", System.Globalization.CultureInfo.InvariantCulture))) + "\n"));

    [Fact]
    public void AColumnThatNeverVaries_HasNoShapeToLookFor()
    {
        // Every candidate leaves a column with no spread at all, so none of them is better than another
        // and the search says so instead of picking whichever came first by an arithmetic accident.
        var table = OneColumn(5, 5, 5, 5);
        var step = new NormaliseStep("a", Scale.Power);

        var learned = step.Fit(table, [.. Enumerable.Repeat(Part.Train, 4)]);

        Assert.Equal(1, learned.Number("lambda"));
        Assert.Equal(1, learned.Number("spread"));
    }

    [Fact]
    public void AColumnSoLargeThatTheShapeOverflows_PassesOverThoseCandidates()
    {
        // Raising a number that size to a power leaves the range of a double entirely. A candidate that
        // produces infinities is not a good fit that happens to be unrepresentable — it is no fit.
        var table = OneColumn(1e200, 2e200, 3e200, 4e200);

        var learned = new NormaliseStep("a", Scale.Power).Fit(table, [.. Enumerable.Repeat(Part.Train, 4)]);

        Assert.True(double.IsFinite(learned.Number("lambda")), "the search returned something unusable");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(2.0)]
    public void TheTwoCandidatesTheFormulaCannotDivideBy_HaveTheirOwnArithmetic(double lambda)
    {
        // At nought and at two the general formula divides by nothing, so the shape is a logarithm there
        // instead. A saved pipeline can name either — the search rounds to four places — and both have to
        // come back to the number that was read.
        var fitted = new FittedStepValues();
        fitted.Learned("lambda", lambda);
        fitted.Learned("centre", 0);
        fitted.Learned("spread", 1);

        var table = OneColumn(3, -3);
        var step = new NormaliseStep("a", Scale.Power);

        step.ApplyTo(table, fitted);
        var shaped = (Column<double>)table["a"];

        Assert.Equal(lambda == 0 ? Math.Log(4) : (Math.Pow(4, 2) - 1) / 2, shaped[0]!.Value, 9);
        Assert.Equal(3, step.Undo(shaped[0]!.Value, fitted), 9);
        Assert.Equal(-3, step.Undo(shaped[1]!.Value, fitted), 9);
    }
}
