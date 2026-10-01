// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the rows keras-fixtures.py and onnx-fixtures.py train the Titanic network on: run once from this folder with
// `dotnet run make-titanic-rows.cs -- <folder>` (.NET 10), then `python keras-fixtures.py <folder>` or
// `python onnx-fixtures.py <folder>`. The rows are not kept
// here — the tests prepare the same rows from the passenger list themselves — and the test project leaves this file out of
// its build.
//
//   train.csv, validation.csv, test.csv   each part of the wiki's Titanic pipeline over Samples/data/titanic.csv, handed
//                                         over as a network takes it: fourteen features on one scale, then `survived`.
//   served.csv                            the two passengers the networks sample serves — a third-class man of 22 and a
//                                         first-class woman of 38 — served through the same fit.
//
// Every number is written as the single-precision float the network is handed, in its shortest exact form.

#:project ../../../../Src/DeepSharp.Pipelines/DeepSharp.Pipelines.csproj

using System.Globalization;
using DeepSharp.Pipelines;

var folder = args.Length > 0 ? args[0] : "rows";
Directory.CreateDirectory(folder);

var prepared = Pdd.Create()
    .ReadCsv(Path.GetFullPath(Path.Join("..", "..", "..", "..", "Samples", "data", "titanic.csv")))
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
    .Run();

static string Written(double value) => ((float)value).ToString("R", CultureInfo.InvariantCulture);

foreach (var part in new[] { Part.Train, Part.Validation, Part.Test })
{
    var batch = prepared.Batch(part, Needs.OneScale);
    var lines = new List<string> { string.Join(",", batch.FeatureNames.Append("survived")) };

    for (var row = 0; row < batch.RowCount; row++)
    {
        lines.Add(string.Join(",", batch.Features[row].Select(Written).Append(Written(batch.Answers![row][0]))));
    }

    File.WriteAllLines(Path.Join(folder, $"{part.ToString().ToLowerInvariant()}.csv"), lines);
    Console.WriteLine($"{part}: {batch.RowCount} rows of {batch.Width} features");
}

var passengers = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]);
var served = prepared.Served(passengers, Needs.OneScale);

File.WriteAllLines(
    Path.Join(folder, "served.csv"),
    [string.Join(",", served.FeatureNames), .. served.Features.Select(row => string.Join(",", row.Select(Written)))]);
Console.WriteLine($"Served: {served.RowCount} passengers");
