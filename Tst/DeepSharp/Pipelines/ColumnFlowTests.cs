// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Which columns there are at each step, known before anything runs. Every step that reads a column says so
/// through its parameters, and every step says what it leaves behind, so a declaration can follow the columns
/// from the schema down: a column the schema left out, or a step above dropped, is refused where it is read —
/// not a whole run later as a column nobody could find. And a front end can offer, at any block, the columns
/// that are there.
/// </summary>
public class ColumnFlowTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema
                .Integer("survived", "sibsp", "parch")
                .Category("sex", "embarked")
                .Optional("age", ColumnKind.Number));

    private static DeclarationFault RefusedAt(Func<object> build) =>
        Assert.Single(Assert.Throws<DeclarationException>(build).Faults);

    [Fact]
    public void AColumnTheSchemaLeftOut_IsRefusedWhereItIsRead_NamingTheStepThatReadsIt()
    {
        // Leaving fare out of the schema used to fail mid-run, as a column nobody could find.
        var fault = RefusedAt(() => Passengers().SplitStratified("survived", 0.70, 0.15).Normalise("fare"));

        Assert.Equal("normalise", fault.Verb);
        Assert.Contains("'fare'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnAnIndicatorReadsInEveryPlace_IsOneRead_AndSoOneFault()
    {
        // Rows with one price may name it as the high, the low and the close; a price that is not there is still
        // one column missing, said once.
        IPipelineStep[] steps =
        [
            new ReadRowsStep("rows"),
            new DeclareStep([new ColumnDeclaration("t", ColumnKind.Integer, Optional: false)]),
            new OrderByStep(["t"]),
            new AddIndicatorStep("range", Indicator.Atr, ["price", "price", "price"]),
        ];

        var fault = Assert.Single(PipelineDeclaration.FaultsIn(steps));

        Assert.Equal("feature.indicator", fault.Verb);
        Assert.Equal([new ColumnRead("price", ColumnKinds.Numbers)], steps[3].ColumnsRead, new ColumnReadComparer());
    }

    // Two reads are the same read when they name the same column and take the same kinds.
    private sealed class ColumnReadComparer : IEqualityComparer<ColumnRead>
    {
        public bool Equals(ColumnRead x, ColumnRead y) => x.Column == y.Column && x.Accepts.SequenceEqual(y.Accepts);

        public int GetHashCode(ColumnRead obj) => obj.Column.GetHashCode(StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnAStepAboveMade_CanBeRead()
    {
        var declaration = Passengers()
            .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("family", "age", "age_was_missing")
            .Declaration;

        Assert.Empty(PipelineDeclaration.FaultsIn(declaration.Steps));
    }

    [Fact]
    public void AHalfOfAValueSplitByItsSign_IsNotScaledOnItsOwn()
    {
        // A scale learned per column after a split by sign would stretch each half by its own extremes and give
        // one quantity two slopes, so the split sign is the last form a value takes, and a scale after it is
        // refused where it is written.
        static FittingBuilder Dated() =>
            Pdd.Create()
                .ReadCsv(Repository.Data("apple.csv"))
                .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
                .Cyclical("Date", Period.DayOfWeek, Form.SplitSign)
                .SplitByTime("Date", 0.70, 0.15);

        var fault = RefusedAt(() => Dated().Normalise("Date_dayofweek_sin_pos"));

        Assert.Equal("normalise", fault.Verb);
        Assert.Contains("'Date_dayofweek_sin_pos'", fault.Message, StringComparison.Ordinal);
        Assert.Contains("'Date_dayofweek_sin'", fault.Message, StringComparison.Ordinal);
        Assert.Empty(PipelineDeclaration.FaultsIn(
            Dated().Normalise("AAPL.Close").FillNaN("Date_dayofweek_cos_neg").Declaration.Steps));
    }

    [Fact]
    public void AColumnStaysAHalf_WhenAStepWritesItAgain_AndNothingElseIsOne()
    {
        var halves = ColumnState.None.WithHalf("x_pos", "x").WithHalf("x_neg", "x");

        Assert.Equal("x", halves.Find("x_pos")!.Value.HalfOf);
        Assert.Equal("x", halves.With("x_pos", ColumnKind.Number).Find("x_pos")!.Value.HalfOf);
        Assert.Null(halves.With("y", ColumnKind.Number).Find("y")!.Value.HalfOf);
        Assert.Null(halves.Without("x_pos").Find("x_pos"));
        Assert.Throws<ArgumentException>(() => ColumnState.None.WithHalf("x_pos", " "));
    }

    [Fact]
    public void AColumnDroppedAbove_CannotBeRead()
    {
        var fault = RefusedAt(() => Passengers().Drop("sibsp").AddFeature("family", "sibsp", Arithmetic.Plus, "parch"));

        Assert.Equal("feature.add", fault.Verb);
        Assert.Contains("'sibsp'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnOfAKindTheStepCannotWorkOn_IsRefusedWhereItIsRead()
    {
        var fault = RefusedAt(() => Passengers().SplitStratified("survived", 0.70, 0.15).Normalise("sex"));

        Assert.Contains("category", fault.Message, StringComparison.Ordinal);
        Assert.Contains("'sex'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnTheSourceLacks_IsRefusedBeforeAnythingRuns_EveryStepThatReadsItAtOnce()
    {
        // The schema allows the column to be absent, so the declaration can say nothing; the rows then say it
        // is, and every step that would read it is named in one refusal before any of them runs.
        var rows = new InMemoryRowSource(["survived", "fare"], [["1", "7.25"], ["0", "8.05"], ["1", "9.5"]]);

        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create()
                .Read(rows, "three passengers")
                .Declare(schema => schema.Integer("survived").Number("fare").Optional("age", ColumnKind.Number))
                .AddFeature("older", "age", Arithmetic.Plus, "fare")
                .SplitAtRandom(0.70)
                .Normalise("age")
                .Build()
                .Run());

        Assert.Equal(["feature.add", "normalise"], refused.Faults.Select(fault => fault.Verb));
        Assert.All(refused.Faults, fault => Assert.Contains("'age'", fault.Message, StringComparison.Ordinal));
    }

    [Fact]
    public void AfterASchemaThatKeepsTheRest_AnyColumnMayBeRead()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Number("a"), Remainder.Keep)
            .AddFeature("c", "a", Arithmetic.Plus, "b")
            .Declaration;

        Assert.Empty(PipelineDeclaration.FaultsIn(declaration.Steps));
        Assert.True(declaration.ColumnsBefore(2).Open);
    }

    [Fact]
    public void AColumnReadIsTakenFromTheStepsParameters_EvenForAStepThatSaysNothingElse()
    {
        // A step from another package says which columns it reads by naming them as columns in its
        // parameters, and nothing more is asked of it. What it leaves behind it does not say, so from there
        // on any column may be there.
        var catalog = StepCatalog.BuiltIn();
        catalog.Register<ScaleByStep>();

        var refused = Assert.Throws<DeclarationException>(() => new PipelineDeclaration([
            new ReadCsvStep("x.csv"),
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]),
            new ScaleByStep("b", 2),
        ]));

        Assert.Equal("scale.by", Assert.Single(refused.Faults).Verb);

        var open = new PipelineDeclaration([
            new ReadCsvStep("x.csv"),
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]),
            new ScaleByStep("a", 2),
            new AddFeatureStep("c", "anything", Arithmetic.Plus, "a"),
        ]);

        Assert.True(open.ColumnsBefore(3).Open);
    }

    [Fact]
    public void AMemberOfAFamilyCanBeRead_AndIsGoneOnceItIsDropped()
    {
        // Which columns an encoder makes is known only once it has seen the training rows, so its family is
        // known by its name alone: sex_ and whatever follows. A member dropped by name is remembered as gone.
        var fitted = Passengers().SplitStratified("survived", 0.70, 0.15).Encode("sex");

        Assert.Empty(PipelineDeclaration.FaultsIn(fitted.Normalise("sex_female").Declaration.Steps));

        var fault = RefusedAt(() => fitted.Drop("sex_male").Normalise("sex_male"));

        Assert.Contains("'sex_male'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnTakenAwayByName_IsGoneEvenWhereAnyOtherColumnMayBe()
    {
        // A schema that keeps the rest leaves any column possible, but not one a step above took away by name.
        var fault = RefusedAt(() => Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Number("a", "b"), Remainder.Keep)
            .Drop("b")
            .AddFeature("c", "a", Arithmetic.Plus, "b"));

        Assert.Contains("'b'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingAfterTheTarget_TakesTheTargetsColumnAway()
    {
        var fault = RefusedAt(() => Passengers().SplitStratified("survived", 0.70, 0.15).Target("survived").Drop("survived"));

        Assert.Equal("drop.columns", fault.Verb);
        Assert.Contains("'survived'", fault.Message, StringComparison.Ordinal);
        Assert.Contains("output", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingAfterTheOutput_TakesAnyOfItsAnswersAway()
    {
        // An output may name several answers — the bins of a histogram — and taking away any one of them leaves a
        // model asked for an answer that is no longer there.
        IPipelineStep[] steps =
        [
            new ReadRowsStep("rows"),
            new DeclareStep([.. new[] { "a", "b", "c" }.Select(name => new ColumnDeclaration(name, ColumnKind.Number, Optional: false))]),
            new NamesTheseAnswers("b", "c"),
            new DropColumnsStep(["c"]),
        ];

        var fault = Assert.Single(PipelineDeclaration.FaultsIn(steps));

        Assert.Equal(3, fault.At);
        Assert.Contains("'c'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EncodingCategoriesThatWereDroppedAbove_IsRefusedWhereItIsWritten()
    {
        var fault = RefusedAt(() => Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Integer("survived").Category("sex"))
            .Drop("sex")
            .SplitAtRandom(0.70, 0.15)
            .EncodeCategories());

        Assert.Contains("no categories", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ColumnsBeforeABlock_AreTheOnesItCanPickFrom()
    {
        var declaration = Passengers()
            .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Declaration;

        var atTheSplit = declaration.ColumnsBefore(3);
        var atTheEnd = declaration.ColumnsBefore(declaration.Steps.Count);

        Assert.Equal(
            ["survived", "sibsp", "parch", "sex", "embarked", "age", "family"],
            atTheSplit.Columns.Select(column => column.Name));
        Assert.False(atTheSplit.Columns.Single(column => column.Name == "age").Surely);
        Assert.Contains("age_was_missing", atTheEnd.Columns.Select(column => column.Name));
        Assert.DoesNotContain("sex", atTheEnd.Columns.Select(column => column.Name));
        Assert.Contains("sex_", atTheEnd.Families);
        Assert.True(atTheEnd.Allows("embarked_S"));
        Assert.Empty(declaration.ColumnsBefore(0).Columns);
        Assert.Throws<ArgumentOutOfRangeException>(() => declaration.ColumnsBefore(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => declaration.ColumnsBefore(declaration.Steps.Count + 1));
    }

    [Fact]
    public void EveryStepThisLibraryShips_SaysWhatItLeavesBehind()
    {
        var undescribed = new[] { typeof(Pdd).Assembly, typeof(AddIndicatorStep).Assembly }
            .SelectMany(assembly => assembly.GetTypes())
            .Where(type => type is { IsClass: true, IsAbstract: false } && typeof(IPipelineStep).IsAssignableFrom(type))
            .Where(type => !typeof(IDescribesColumns).IsAssignableFrom(type))
            .Select(type => type.Name)
            .ToArray();

        Assert.Empty(undescribed);
    }

    public static TheoryData<string> EveryVerb()
    {
        var verbs = new TheoryData<string>();

        foreach (var description in StepCatalog.BuiltIn().WithIndicators().Descriptions)
        {
            verbs.Add(description.Verb);
        }

        return verbs;
    }

    [Theory]
    [MemberData(nameof(EveryVerb))]
    public void EveryStepDoesToTheColumnsWhatItSaysItDoes(string verb)
    {
        // The description and the act are two things that could drift, so each step is run on a table and the
        // columns it leaves behind are held to the ones it said it would, a family matched by its name.
        var table = SchemaBinding.Bind(
            new DeclareStep([
                new ColumnDeclaration("when", ColumnKind.Timestamp, false),
                new ColumnDeclaration("a", ColumnKind.Number, true),
                new ColumnDeclaration("b", ColumnKind.Integer, false),
                new ColumnDeclaration("g", ColumnKind.Category, false),
                new ColumnDeclaration("w", ColumnKind.Text, false),
            ]),
            new InMemoryRowSource(
                ["when", "a", "b", "g", "w"],
                [.. Enumerable.Range(0, 40).Select(row => (IReadOnlyList<string?>)
                [
                    new DateTime(2020, 1, 1).AddHours(row * 7).ToString("o", System.Globalization.CultureInfo.InvariantCulture),
                    row % 5 == 0 ? null : (row * 1.5).ToString(System.Globalization.CultureInfo.InvariantCulture),
                    (row % 7).ToString(System.Globalization.CultureInfo.InvariantCulture),
                    row % 3 == 0 ? "x" : "y",
                    "words",
                ])]));

        var step = StepFor(verb);
        var before = ColumnState.Of(table);

        // A schema's act is to read rows into its columns, from nothing that was there before.
        if (step is DeclareStep declare)
        {
            before = ColumnState.Of(new Table([]));
            table = declare.Bind(new InMemoryRowSource([.. declare.Columns.Select(column => column.Name)], [[.. declare.Columns.Select(_ => (string?)"1")]]));
        }

        var said = ((IDescribesColumns)step).After(before);

        Act(step, table);

        // Every column it said, exactly; and every other column it made belongs to a family it said.
        Assert.Equal(
            said.Columns.Select(column => $"{column.Name}:{column.Kind}").Order(StringComparer.Ordinal),
            table.Columns.Where(column => said.Find(column.Name) is not null || !said.InAFamily(column.Name))
                .Select(column => $"{column.Name}:{column.Kind}").Order(StringComparer.Ordinal));

        Assert.All(
            table.Columns.Where(column => said.Find(column.Name) is null),
            column => Assert.True(said.InAFamily(column.Name), $"'{column.Name}' belongs to no family '{verb}' said it makes."));
    }

    private static IPipelineStep StepFor(string verb)
    {
        var catalog = StepCatalog.BuiltIn().WithIndicators();

        return verb switch
        {
            "feature.add" => new AddFeatureStep("sum", "a", Arithmetic.Plus, "b"),
            "feature.cyclical" => new CyclicalStep("when", Period.HourOfDay, Form.SplitSign),
            "feature.timeParts" => new TimePartsStep("when", [TimePart.Hour, TimePart.Season]),
            "feature.indicator" => new AddIndicatorStep("bands", Indicator.BollingerBands, ["b"], 3),
            "fill.missing" => FillMissingStep.Of("a", With.Median),
            "fill.nan" => new FillNaNStep("a", With.Zero),
            "maths" => new MathsStep("b", Maths.Square, "b2"),
            "normalise" => new NormaliseStep("b"),
            "normalise.row" => new NormaliseRowStep(["a", "b"]),
            "encode" => new EncodeStep("g"),
            "encode.categories" => new EncodeCategoriesStep(As.Ordinal),
            "outliers.clip" => new ClipOutliersStep("b"),
            "drop.columns" => new DropColumnsStep(["w"]),
            "order.by" => new OrderByStep(["when"]),
            "split.byTime" => new SplitByTimeStep("when", new SplitShares(0.70, 0.15, 0.15)),
            "split.atRandom" => new SplitAtRandomStep(new SplitShares(0.70, 0.15, 0.15), 1),
            "split.stratified" => new SplitStratifiedStep("g", new SplitShares(0.70, 0.15, 0.15), 1),
            "target" => new TargetStep("b"),
            "drop.warmup" => new DropWarmUpStep(),
            _ => catalog.Read(JsonDocument.Parse(catalog.Describe(verb).Template).RootElement),
        };
    }

    private static void Act(IPipelineStep step, Table table)
    {
        var parts = Enumerable.Range(0, table.RowCount).Select(row => row % 4 == 0 ? Part.Test : Part.Train).ToArray();

        switch (step)
        {
            case IAddsColumns adds:
                adds.AddTo(table);
                break;

            case IFittedStep learns:
                learns.ApplyTo(table, learns.Fit(table, parts));
                break;

            case IDropsColumns drops:
                drops.DropFrom(table);
                break;
        }
    }
}
