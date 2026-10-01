// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the rows pytorch.py trains PyTorch's networks on: run once from this folder with `dotnet run make-rows.cs`
// (.NET 10), then pytorch.py, and commit what both write beside them. The test project leaves this file out of its build.
//
// The rows are the Titanic passenger list prepared by the wiki's pipeline, as it hands them to a network — every feature
// on one scale — each value written as the 32-bit float a network here reads, so PyTorch is handed the very numbers:
//
//   titanic-train.csv        the training rows and whether each passenger survived
//   titanic-validation.csv   the validation rows, which PyTorch stops early on
//   titanic-test.csv         the test rows
//   titanic-served.csv       the README's passenger, a man of 22 in third class, and a woman of 38 in first, served

#:project ../../../../Src/DeepSharp.Pipelines/DeepSharp.Pipelines.csproj

using System.Globalization;
using DeepSharp.Pipelines;

var prepared = Pdd.Create()
    .ReadCsv(Path.Join("..", "..", "..", "..", "Samples", "data", "titanic.csv"))
    .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
    .SplitStratified("survived", train: 0.70, validation: 0.15)
    .FillMissing("age", With.Median)
    .EncodeCategories()
    .Normalise("age", Scale.MidRange)
    .Normalise("fare", Scale.MidRange)
    .Normalise("sibsp", Scale.MidRange)
    .Normalise("parch", Scale.MidRange)
    .Target("survived")
    .Build()
    .Run();

static string Number(double value) => ((float)value).ToString("R", CultureInfo.InvariantCulture);

foreach (var part in new[] { Part.Train, Part.Validation, Part.Test })
{
    var batch = prepared.Batch(part, Needs.OneScale);
    var lines = new List<string> { string.Join(',', batch.FeatureNames.Append("survived")) };

    for (var row = 0; row < batch.RowCount; row++)
    {
        lines.Add(string.Join(',', batch.Features[row].Select(Number).Append(Number(batch.Answers![row][0]))));
    }

    File.WriteAllText($"titanic-{part.ToString().ToLowerInvariant()}.csv", string.Join('\n', lines) + "\n");
    Console.WriteLine($"{part}: {batch.RowCount} rows of {batch.Width}");
}

var served = prepared.Served(
    new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]),
    Needs.OneScale);

File.WriteAllText(
    "titanic-served.csv",
    string.Join('\n', served.Features.Select(row => string.Join(',', row.Select(Number))).Prepend(string.Join(',', served.FeatureNames))) + "\n");
Console.WriteLine($"Served: {served.RowCount} rows of {served.FeatureNames.Count}");
