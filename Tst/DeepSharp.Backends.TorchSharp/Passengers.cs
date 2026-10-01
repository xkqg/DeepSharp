// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// The published passenger list as the networks sample prepares it: read, declared, divided by who survived, the age filled,
/// every category one-hot, four scales onto minus one to one, the answer, and the report the sample declares.
/// </summary>
internal static class Passengers
{
    /// <summary>The pipeline, run over the published list.</summary>
    public static PreparedData Prepared() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
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
                .As(Shown.Numbers))
            .Build()
            .Run();

    /// <summary>The two passengers the sample serves after training: a third-class man of 22 and a first-class woman of 38.</summary>
    public static InMemoryRowSource Served() =>
        new(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]);
}
