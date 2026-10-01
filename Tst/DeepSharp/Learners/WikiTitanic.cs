// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The wiki's Titanic pipeline, as 0.4.0 ran it and as the files 0.4.0 wrote carry it: read, declared, divided by who
/// survived, the age filled, every category one-hot, four scales onto minus one to one, the answer, and the report.
/// </summary>
internal static class WikiTitanic
{
    /// <summary>The steps, reading <c>titanic.csv</c> from where the pipeline stands.</summary>
    public static PipelineDeclaration Declaration { get; } =
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Report(report => report
                .Measure(Metric.Accuracy, Metric.Precision, Metric.Recall, Metric.ConfusionMatrix)
                .On(Part.Train, Part.Validation, Part.Test)
                .As(Shown.Numbers, Shown.Drawn))
            .Build()
            .Declaration;

    /// <summary>The folder the published passenger list stands in.</summary>
    public static string DataFolder => Path.GetDirectoryName(Repository.Data("titanic.csv"))!;

    /// <summary>The pipeline, reading its passenger list from a folder.</summary>
    /// <param name="folder">The folder that holds a <c>titanic.csv</c>.</param>
    /// <returns>The pipeline.</returns>
    public static Pipeline In(string folder) => new(Declaration, rows: null, SourceFolder.Of(folder));

    /// <summary>The passenger on a line of the published list, read as the list writes it, under its header.</summary>
    /// <param name="line">The line, counting from one, the header being the first.</param>
    /// <returns>The one row.</returns>
    public static CsvRowSource Line(int line)
    {
        var lines = File.ReadLines(Repository.Data("titanic.csv")).ToArray();

        return CsvRowSource.FromText($"{lines[0]}\n{lines[line - 1]}\n");
    }
}
