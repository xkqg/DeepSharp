// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a trained model has to prove is declared with its pipeline, before any number exists: which measures, on
/// which parts, shown how. The report names them and does nothing a run acts on — the measures are taken once a
/// model has predicted — so it is admitted as the one other step, beside an output, that acts on nothing. It stands
/// below the output whose answers it measures, there is one of it, it measures the parts a split makes and no
/// other rows, and a measure that counts classes is refused against an output whose answers are amounts.
/// </summary>
public class ReportTests
{
    private static readonly SplitShares Shares = new(0.70, 0.15, 0.15);

    private static DeclareStep Schema(params string[] names) =>
        new([.. names.Select(name => new ColumnDeclaration(name, ColumnKind.Number, false))]);

    private static ReportStep Report(params Metric[] metrics) =>
        new(metrics.Length == 0 ? [Metric.Rmse] : metrics, [Part.Validation, Part.Test], [Shown.Numbers]);

    // A source, three columns of numbers, a split at random, and whatever follows.
    private static IPipelineStep[] Steps(params IPipelineStep[] below) =>
        [new ReadCsvStep("a.csv"), Schema("a", "b", "c"), new SplitAtRandomStep(Shares, 1), .. below];

    private static FittingBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories();

    [Fact]
    public void AReportBelowTheOutput_IsAStepThePipelineTakes_ThoughTheRunActsOnNothingForIt()
    {
        var report = Report();
        var steps = Steps(new TargetStep("a"), report);

        Assert.Empty(PipelineDeclaration.FaultsIn(steps));
        Assert.Same(report, new PipelineDeclaration(steps).Report);
        Assert.Null(new PipelineDeclaration(Steps(new TargetStep("a"))).Report);
        Assert.IsNotAssignableFrom<IActsInAWalk>(report);
    }

    [Fact]
    public void ASecondReport_IsRefusedAtTheSecond()
    {
        var fault = Assert.Single(PipelineDeclaration.FaultsIn(Steps(new TargetStep("a"), Report(), Report(Metric.Mae))));

        Assert.Equal(5, fault.At);
        Assert.Equal("evidence.report", fault.Verb);
        Assert.Contains("one report", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AReportAboveTheOutput_IsRefusedAtTheReport()
    {
        var fault = Assert.Single(PipelineDeclaration.FaultsIn(Steps(Report(), new TargetStep("a"))));

        Assert.Equal(3, fault.At);
        Assert.Contains("step 5", fault.Message, StringComparison.Ordinal);
        Assert.Contains("below the output", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AReportWithNoOutput_IsRefused()
    {
        var fault = Assert.Single(PipelineDeclaration.FaultsIn(Steps(Report())));

        Assert.Equal(3, fault.At);
        Assert.Contains("names no answer", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AReportInAPipelineThatNeverDividesItsRows_IsRefused()
    {
        // There are no parts to measure on: every row is undivided, and none is a row a model learns from, is chosen on
        // or is tested on.
        var fault = Assert.Single(PipelineDeclaration.FaultsIn(
            [new ReadCsvStep("a.csv"), Schema("a", "b"), new TargetStep("a"), Report()]));

        Assert.Equal(3, fault.At);
        Assert.Contains("never divides", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Part.Predict, "predict")]
    [InlineData(Part.Gap, "gap")]
    [InlineData(Part.Undivided, "undivided")]
    public void AReportOnRowsNothingIsMeasuredOn_IsRefusedWhereItIsWritten(Part part, string word)
    {
        var inCode = Assert.ThrowsAny<ArgumentException>(() => new ReportStep([Metric.Rmse], [Part.Test, part], [Shown.Numbers]));

        Assert.Contains("'parts'", inCode.Message, StringComparison.Ordinal);
        Assert.Contains("train, validation, test", inCode.Message, StringComparison.Ordinal);
        Assert.ThrowsAny<ArgumentException>(() => Passengers().Target("survived").Report(report => report.Measure(Metric.Rmse).On(part).As(Shown.Numbers)));

        var file = $$"""
            {"version": 3, "declaration": [
              {"step": "read.csv", "path": "a.csv"},
              {"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}]},
              {"step": "split.atRandom", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 1},
              {"step": "target", "column": "a"},
              {"step": "evidence.report", "metrics": ["rmse"], "parts": ["test", "{{word}}"], "shown": ["numbers"]}
            ]}
            """;

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(file, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(6, fault.Line);
        Assert.Contains($"'{word}'", fault.Message, StringComparison.Ordinal);
        Assert.Contains("train, validation, test", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Metric.Accuracy)]
    [InlineData(Metric.Precision)]
    [InlineData(Metric.Recall)]
    [InlineData(Metric.ConfusionMatrix)]
    public void AMeasureThatCountsClasses_IsRefusedWhereItIsWritten_AgainstSharesOfAWholeOrAReturn(Metric metric)
    {
        var shares = Assert.Single(PipelineDeclaration.FaultsIn(Steps(new DistributionStep(["b", "c"]), Report(Metric.Rmse, metric))));

        Assert.Equal(4, shares.At);
        Assert.Contains("target.distribution", shares.Message, StringComparison.Ordinal);
        Assert.Contains(metric.ToString().ToLowerInvariant(), shares.Message, StringComparison.Ordinal);

        var refused = Assert.Throws<DeclarationException>(() => AheadSplit().Ahead("close", 5, AheadAs.Return).Report(
            report => report.Measure(metric).On(Part.Test).As(Shown.Numbers)));

        Assert.Equal("evidence.report", Assert.Single(refused.Faults).Verb);

        // Classes are counted wherever the answers can be classes: a column of noughts and ones, labels, a value read ahead.
        Assert.Empty(PipelineDeclaration.FaultsIn(Steps(new TargetStep("a"), Report(metric))));
        Assert.Empty(PipelineDeclaration.FaultsIn(Steps(new LabelsStep(["b", "c"], ones: 1), Report(metric))));
        Assert.NotNull(AheadSplit().Ahead("close", 5).Report(report => report.Measure(metric).On(Part.Test).As(Shown.Numbers)).Declaration.Report);
    }

    [Fact]
    public void EveryMeasureThatCountsClasses_IsNamedInTheOneFault()
    {
        var fault = Assert.Single(PipelineDeclaration.FaultsIn(Steps(
            new DistributionStep(["b", "c"]), Report(Metric.Accuracy, Metric.Rmse, Metric.Recall, Metric.Accuracy, Metric.Precision))));

        Assert.Contains("counts classes with accuracy, recall and precision, and", fault.Message, StringComparison.Ordinal);
        Assert.Contains("rmse, mae or r2", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AmountsAreMeasuredForEveryKindOfOutput()
    {
        Assert.Empty(PipelineDeclaration.FaultsIn(Steps(new DistributionStep(["b", "c"], scaleBy: "a"), Report(Metric.Rmse, Metric.Mae, Metric.R2))));
        Assert.All(
            new INamesTheAnswer[] { new TargetStep("a"), new DistributionStep(["b", "c"]), new LabelsStep(["b", "c"]), new NumbersStep(["b", "c"]), new AheadStep("close", 5) },
            output => Assert.True(output.Takes(MetricFamily.Amounts)));
        Assert.NotNull(AheadSplit().Ahead("close", 5, AheadAs.Return).Report(
            report => report.Measure(Metric.Rmse, Metric.Mae, Metric.R2).On(Part.Train, Part.Test).As(Shown.Drawn)).Declaration.Report);
    }

    [Fact]
    public void AnOutputSaysWhetherItsAnswersCanBeClasses_AndHowManyOfThemARowHolds()
    {
        INamesTheAnswer target = new TargetStep("a");
        INamesTheAnswer shares = new DistributionStep(["b", "c"]);
        INamesTheAnswer one = new LabelsStep(["b", "c"], ones: 1);
        INamesTheAnswer any = new LabelsStep(["b", "c"]);
        INamesTheAnswer value = new AheadStep("close", 5);
        INamesTheAnswer ret = new AheadStep("close", 5, AheadAs.Return);

        Assert.Equal([true, false, true, true, true, false], new[] { target, shares, one, any, value, ret }.Select(output => output.AnswersCanBeClasses));
        Assert.Equal([0, 0, 1, 0, 0, 0], new[] { target, shares, one, any, value, ret }.Select(output => output.Ones));
    }

    [Fact]
    public void AReport_SurvivesTheFile()
    {
        var declaration = Passengers()
            .Target("survived")
            .Report(report => report
                .Measure(Metric.Accuracy, Metric.ConfusionMatrix, Metric.Rmse)
                .On(Part.Train, Part.Validation, Part.Test)
                .As(Shown.Numbers, Shown.Drawn))
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());
        var report = Assert.IsType<ReportStep>(returned.Report);

        Assert.Equal(declaration, returned);
        Assert.Equal([Metric.Accuracy, Metric.ConfusionMatrix, Metric.Rmse], report.Metrics);
        Assert.Equal([Part.Train, Part.Validation, Part.Test], report.Parts);
        Assert.Equal([Shown.Numbers, Shown.Drawn], report.Shown);
        Assert.Contains(
            """{"step":"evidence.report","metrics":["accuracy","confusionmatrix","rmse"],"parts":["train","validation","test"],"shown":["numbers","drawn"]}""",
            returned.ToJson().Replace("\n", string.Empty, StringComparison.Ordinal).Replace(" ", string.Empty, StringComparison.Ordinal),
            StringComparison.Ordinal);
    }

    [Fact]
    public void AReport_InAFileOfTheSecondVersion_IsRefusedByName()
    {
        // The step is newer than the second version of the file: an older library would not know it, so a file that
        // says it was written against that version and names it was not written by any library that meant it.
        const string file = """
            {"version": 2, "declaration": [
              {"step": "read.csv", "path": "a.csv"},
              {"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}]},
              {"step": "split.atRandom", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 1},
              {"step": "target", "column": "a"},
              {"step": "evidence.report", "metrics": ["rmse"], "parts": ["test"], "shown": ["numbers"]}
            ]}
            """;

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(file, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(6, fault.Line);
        Assert.Contains("version 3", fault.Message, StringComparison.Ordinal);
        Assert.Contains("version 2", fault.Message, StringComparison.Ordinal);
        Assert.Equal(3, StepCatalog.BuiltIn().Describe("evidence.report").Since);
    }

    [Fact]
    public void TheChain_WritesTheReportItIsTold_EachWordAddingToWhatWasSaid()
    {
        var builder = new ReportBuilder().Measure(Metric.Rmse).Measure(Metric.Mae).On(Part.Test).As(Shown.Drawn);

        Assert.Equal([Metric.Rmse, Metric.Mae], builder.Metrics);
        Assert.Equal([Part.Test], builder.Parts);
        Assert.Equal([Shown.Drawn], builder.Shown);

        var declaration = Passengers()
            .Target("survived")
            .Report(report => report.Measure(Metric.Rmse).Measure(Metric.Mae).On(Part.Test).As(Shown.Drawn))
            .Declaration;

        Assert.Equal(new ReportStep([Metric.Rmse, Metric.Mae], [Part.Test], [Shown.Drawn]), declaration.Steps[^1]);
        Assert.Throws<ArgumentNullException>(() => Passengers().Report(null!));
    }

    [Fact]
    public void AReportThatMeasuresNothing_OnNothing_OrShowsNothing_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new ReportStep([], [Part.Test], [Shown.Numbers]));
        Assert.Throws<ArgumentException>(() => new ReportStep([Metric.Rmse], [], [Shown.Numbers]));
        Assert.Throws<ArgumentException>(() => new ReportStep([Metric.Rmse], [Part.Test], []));
        Assert.Throws<ArgumentNullException>(() => new ReportStep(null!, [Part.Test], [Shown.Numbers]));
        Assert.Throws<ArgumentNullException>(() => new ReportStep([Metric.Rmse], null!, [Shown.Numbers]));
        Assert.Throws<ArgumentNullException>(() => new ReportStep([Metric.Rmse], [Part.Test], null!));

        // A chain that forgets to say how the measures are shown is refused where it is written, not at the run.
        var refused = Assert.Throws<ArgumentException>(() => Passengers().Target("survived").Report(report => report.Measure(Metric.Rmse).On(Part.Test)));

        Assert.Contains("'shown'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoReportsThatSayTheSameThing_AreTheSameStep()
    {
        Assert.Equal(Report(Metric.Rmse, Metric.R2), Report(Metric.Rmse, Metric.R2));
        Assert.Equal(Report(Metric.Rmse, Metric.R2).GetHashCode(), Report(Metric.Rmse, Metric.R2).GetHashCode());
        Assert.NotEqual(Report(Metric.Rmse, Metric.R2), Report(Metric.R2, Metric.Rmse));
        Assert.NotEqual(Report(), new ReportStep([Metric.Rmse], [Part.Test], [Shown.Numbers]));
        Assert.NotEqual(Report(), new ReportStep([Metric.Rmse], [Part.Validation, Part.Test], [Shown.Drawn]));
        Assert.False(Report().Equals(null));
        Assert.Equal("evidence.report", Report().Verb);
    }

    [Fact]
    public void TheRunLeavesAReportAlone_AndProducesNoEvidenceForIt()
    {
        var without = Passengers().Target("survived").Build().Run();
        var with = Passengers().Target("survived").Report(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers)).Build().Run();

        Assert.Equal(without.Table.Columns.Select(column => column.Name), with.Table.Columns.Select(column => column.Name));
        Assert.Equal(without.Parts, with.Parts);
        Assert.Equal(without.Fitted.Keys, with.Fitted.Keys);
        Assert.Empty(with.Evidence);

        // A replay walks the same steps and passes the report by as well.
        var replayed = with.Replay(new InMemoryRowSource(["survived", "pclass", "sex"], [["1", "3", "female"]]));

        Assert.Equal(1, replayed.RowCount);
    }

    [Fact]
    public void ThePartsAReportTakes_AreTheThreeAModelIsTrainedChosenAndTestedOn()
    {
        var description = StepCatalog.BuiltIn().Describe("evidence.report");
        var parts = Assert.IsType<SeveralOfParameter<Part>>(description.Parameters.Single(parameter => parameter.Key == "parts"));
        var metrics = Assert.IsType<SeveralOfParameter<Metric>>(description.Parameters.Single(parameter => parameter.Key == "metrics"));
        var shown = Assert.IsType<SeveralOfParameter<Shown>>(description.Parameters.Single(parameter => parameter.Key == "shown"));

        Assert.Equal(["train", "validation", "test"], parts.Choices);
        Assert.Equal(["rmse", "mae", "r2", "accuracy", "precision", "recall", "confusionmatrix", "emd", "kl", "rps"], metrics.Choices);
        Assert.Equal(["drawn", "numbers"], shown.Choices);
        Assert.Equal(["metrics", "parts", "shown"], description.Parameters.Select(parameter => parameter.Key));

        using var template = JsonDocument.Parse(description.Template);

        Assert.Equal(Report(), StepCatalog.BuiltIn().Read(template.RootElement));
    }

    private static FittingBuilder AheadSplit() =>
        Pdd.Create()
            .Read(new InMemoryRowSource(["Date", "close"], [["2026-01-01", "1"]]), "one day")
            .Declare(schema => schema.Timestamp("Date").Number("close"))
            .OrderBy("Date")
            .SplitByTime("Date", 0.70, 0.15, 5)
            .Drop("Date");
}
