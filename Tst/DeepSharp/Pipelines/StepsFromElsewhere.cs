// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A verb another package brings: a column multiplied by a number, written the way a package author writes a
/// step — from the parameter kinds, with nothing internal.
/// </summary>
public sealed record ScaleByStep : IPipelineStep<ScaleByStep>, IAddsColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column to multiply.", "column", ColumnKinds.Numbers);

    private static readonly NumberParameter ByKey = new("by", "What every value is multiplied by: a number above nothing.", 2) { Above = 0 };

    public ScaleByStep(string column, double by)
    {
        Column = ColumnKey.Require(column);
        By = ByKey.Require(by);
    }

    public string Column { get; }

    public double By { get; }

    public static string Name => "scale.by";

    public static string Purpose => "Multiplies a column by a number.";

    public static StepParameters<ScaleByStep> Parameters { get; } = new StepParameters<ScaleByStep>()
        .With(ColumnKey, step => step.Column)
        .With(ByKey, step => step.By);

    public string Verb => Name;

    public static ScaleByStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element), ByKey.Read(element));

    public void AddTo(Table table) =>
        table.Put(new Column<double>(
            Column, ColumnKind.Number, table.NumbersOf(Column).Select(value => value * By)));
}

/// <summary>A source another package brings, which has nothing to open in a test.</summary>
public sealed record ReadAvroStep : IPipelineStep<ReadAvroStep>, IOpensRows
{
    private static readonly FilePathParameter PathKey = new("path", "Where the file is.", "data.avro");

    public ReadAvroStep(string path) => Path = PathKey.Require(path);

    public string Path { get; }

    public static string Name => "read.avro";

    public static string Purpose => "Reads the rows from an Avro file.";

    public static StepParameters<ReadAvroStep> Parameters { get; } =
        new StepParameters<ReadAvroStep>().With(PathKey, step => step.Path);

    public string Verb => Name;

    public static ReadAvroStep ReadFrom(JsonElement element) => new(PathKey.Read(element));

    public IRowSource Open(SourceFolder folder) => new InMemoryRowSource(["a"], []);
}

/// <summary>A step whose type names one verb while its instances answer to another.</summary>
public sealed record MisnamedStep : IPipelineStep<MisnamedStep>, IOpensRows
{
    public static string Name => "read.parquet";

    public static string Purpose => "Claims to read a Parquet file.";

    public static StepParameters<MisnamedStep> Parameters { get; } = new();

    public string Verb => "read.csv";

    public static MisnamedStep ReadFrom(JsonElement element) => new();

    public IRowSource Open(SourceFolder folder) => new InMemoryRowSource(["a"], []);
}

/// <summary>A step type that never says what it is called.</summary>
public sealed record NamelessStep : IPipelineStep<NamelessStep>, IOpensRows
{
    public static string Name => "  ";

    public static string Purpose => "Has no name.";

    public static StepParameters<NamelessStep> Parameters { get; } = new();

    public string Verb => Name;

    public static NamelessStep ReadFrom(JsonElement element) => new();

    public IRowSource Open(SourceFolder folder) => new InMemoryRowSource(["a"], []);
}

/// <summary>A step type that never says what it does.</summary>
public sealed record PurposelessStep : IPipelineStep<PurposelessStep>, IOpensRows
{
    public static string Name => "read.nothing";

    public static string Purpose => " ";

    public static StepParameters<PurposelessStep> Parameters { get; } = new();

    public string Verb => Name;

    public static PurposelessStep ReadFrom(JsonElement element) => new();

    public IRowSource Open(SourceFolder folder) => new InMemoryRowSource(["a"], []);
}

/// <summary>
/// A step from elsewhere that learns an empty list of words and an empty run of numbers, and on replay asks
/// for both by the kind it learned them as.
/// </summary>
public sealed record LearnNothingStep : IPipelineStep<LearnNothingStep>, IFittedStep
{
    public static string Name => "learn.nothing";

    public static string Purpose => "Learns an empty list and an empty run.";

    public static StepParameters<LearnNothingStep> Parameters { get; } = new();

    public string Verb => Name;

    public static LearnNothingStep ReadFrom(JsonElement element) => new();

    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        var learned = new FittedStepValues();
        learned.Learned("words", Array.Empty<string>());
        learned.Learned("run", Array.Empty<double>());

        return learned;
    }

    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        if (fitted.List("words").Count + fitted.Curve("run").Count != 0)
        {
            throw new InvalidOperationException("Nothing was learned, and something came back.");
        }
    }
}

/// <summary>A step from elsewhere that takes a column off the table and does not say what it leaves behind.</summary>
public sealed record ForgetStep : IPipelineStep<ForgetStep>, IDropsColumns
{
    private static readonly ColumnParameter ColumnKey = new("column", "The column to forget.", "column", ColumnKinds.Any);

    public ForgetStep(string column) => Column = ColumnKey.Require(column);

    public string Column { get; }

    public static string Name => "column.forget";

    public static string Purpose => "Forgets a column.";

    public static StepParameters<ForgetStep> Parameters { get; } =
        new StepParameters<ForgetStep>().With(ColumnKey, step => step.Column);

    public string Verb => Name;

    public static ForgetStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element));

    public void DropFrom(Table table) => table.Remove(Column);
}

/// <summary>
/// A step from elsewhere that a learner indifferent to scale does without — a column less its training mean — and that
/// offers in its place a step written as another verb, which no run can name in its file as the step it stands for.
/// </summary>
public sealed record CentreStep : IPipelineStep<CentreStep>, IMeetsANeed
{
    private static readonly ColumnParameter ColumnKey = new("column", "The column to centre.", "column", ColumnKinds.Numbers);

    public CentreStep(string column) => Column = ColumnKey.Require(column);

    public string Column { get; }

    public static string Name => "centre";

    public static string Purpose => "Takes the training rows' mean off a column.";

    public static StepParameters<CentreStep> Parameters { get; } =
        new StepParameters<CentreStep>().With(ColumnKey, step => step.Column);

    public string Verb => Name;

    IFittedStep? IMeetsANeed.Instead => new LearnNothingStep();

    public static CentreStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element));

    public bool NeededBy(Needs needs) => !needs.DoesWithoutScaling();

    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        var learned = new FittedStepValues();
        learned.Learned("mean", table.NumbersOf(Column).Where((_, row) => parts[row] == Part.Train).Average() ?? 0);

        return learned;
    }

    public void ApplyTo(Table table, FittedStepValues fitted) =>
        table.Put(new Column<double>(Column, ColumnKind.Number, table.NumbersOf(Column).Select(value => value - fitted.Number("mean"))));
}

/// <summary>
/// A step from elsewhere that says every learner does without it, and that its column lands between minus one and one once
/// it has squashed it there: left out, the column lands nowhere said.
/// </summary>
public sealed record SquashStep : IPipelineStep<SquashStep>, IMeetsANeed, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new("column", "The column to squash.", "column", ColumnKinds.Numbers);

    public SquashStep(string column) => Column = ColumnKey.Require(column);

    public string Column { get; }

    public static string Name => "squash";

    public static string Purpose => "Squashes a column between minus one and one.";

    public static StepParameters<SquashStep> Parameters { get; } =
        new StepParameters<SquashStep>().With(ColumnKey, step => step.Column);

    public string Verb => Name;

    IFittedStep? IMeetsANeed.Instead => null;

    public static SquashStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element));

    public bool NeededBy(Needs needs) => false;

    public ColumnState After(ColumnState before) => before.With(Column, ColumnKind.Number, Form.Signed);

    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts) => new();

    public void ApplyTo(Table table, FittedStepValues fitted) =>
        table.Put(new Column<double>(Column, ColumnKind.Number, table.NumbersOf(Column).Select(value => value is { } number ? Math.Tanh(number) : (double?)null)));
}

/// <summary>
/// A step from elsewhere that names the learner the pipeline is declared for: the stump of these tests, which takes
/// numbers of any size. It changes no column and runs nothing — the package that brings it trains once the pipeline has
/// run — so it says which columns it leaves behind, as every no-act step does.
/// </summary>
public sealed record LearnStumpStep : IPipelineStep<LearnStumpStep>, INamesTheLearner, IDescribesColumns
{
    private static readonly WholeNumberParameter DepthKey = new("depth", "How many questions deep the stump asks.", 1) { AtLeast = 1 };

    public LearnStumpStep(int depth) => Depth = DepthKey.Require(depth);

    public int Depth { get; }

    public static string Name => "learn.stump";

    public static string Purpose => "Names a stump as the learner this pipeline is declared for.";

    public static StepParameters<LearnStumpStep> Parameters { get; } =
        new StepParameters<LearnStumpStep>().With(DepthKey, step => step.Depth);

    public string Verb => Name;

    public Needs Needs => Needs.NoScale;

    public static LearnStumpStep ReadFrom(JsonElement element) => new(DepthKey.Read(element));

    public ColumnState After(ColumnState before) => before;
}

/// <summary>A step that names a learner and says nothing of the columns it leaves behind: what the rule above it refuses.</summary>
public sealed record LearnQuietlyStep : IPipelineStep<LearnQuietlyStep>, INamesTheLearner
{
    public static string Name => "learn.quietly";

    public static string Purpose => "Names a learner without saying which columns it leaves behind.";

    public static StepParameters<LearnQuietlyStep> Parameters { get; } = new();

    public string Verb => Name;

    public Needs Needs => Needs.NoScale;

    public static LearnQuietlyStep ReadFrom(JsonElement element) => new();
}
