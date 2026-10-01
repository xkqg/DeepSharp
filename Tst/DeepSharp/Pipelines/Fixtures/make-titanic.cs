// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the Parquet file and the JSON file the readers' tests read: run once from this folder with `dotnet run
// make-titanic.cs` (.NET 10), and commit what it writes beside it. The test project leaves this file out of its build.
//
//   titanic.parquet   Samples/data/titanic.csv, read by DeepSharp's own reader of comma-separated files, every cell as the
//                     CSV holds it. survived, pclass, sibsp and parch are 64-bit whole numbers and adult_male and alone
//                     true or false, since the reader hands those back as the CSV writes them; every other column is
//                     text, as the CSV spells it — 22.0, 71.2833 — since a number would be handed back in its own
//                     spelling, 22. An empty CSV cell is empty text.
//   titanic.json      the same cells as an array of records, one object a row with every column in the CSV's order. A
//                     cell of a column of numbers is a JSON number written exactly as the CSV writes it, 22.0 as 22.0;
//                     every other cell, and an empty one, is a JSON string.

#:package Parquet.Net@6.1.0
#:project ../../../../Src/DeepSharp.Pipelines/DeepSharp.Pipelines.csproj

using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;
using Parquet;
using Parquet.Schema;

var csv = new CsvRowSource(Path.Join("..", "..", "..", "..", "Samples", "data", "titanic.csv"));
var names = csv.ColumnNames;
var rows = csv.Rows.ToArray();
string[] wholeNumbers = ["survived", "pclass", "sibsp", "parch"];
string[] truths = ["adult_male", "alone"];
string[] numbers = ["survived", "pclass", "age", "sibsp", "parch", "fare"];

string[] Cells(int column) => [.. rows.Select(row => row[column]!)];

var fields = names.Select(name => wholeNumbers.Contains(name)
        ? new DataField<long>(name)
        : truths.Contains(name) ? new DataField<bool>(name) : (DataField)new DataField<string>(name))
    .ToArray();

using (var file = File.Create("titanic.parquet"))
{
    await using var writer = await ParquetWriter.CreateAsync(new ParquetSchema(fields), file);
    using var group = writer.CreateRowGroup();

    for (var column = 0; column < names.Count; column++)
    {
        if (wholeNumbers.Contains(names[column]))
        {
            await group.WriteAsync<long>(fields[column], Cells(column).Select(long.Parse).ToArray());
        }
        else if (truths.Contains(names[column]))
        {
            await group.WriteAsync<bool>(fields[column], Cells(column).Select(bool.Parse).ToArray());
        }
        else
        {
            await group.WriteAsync(fields[column], Cells(column));
        }
    }
}

using (var file = File.Create("titanic.json"))
{
    using var writer = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });

    writer.WriteStartArray();

    foreach (var row in rows)
    {
        writer.WriteStartObject();

        for (var column = 0; column < names.Count; column++)
        {
            writer.WritePropertyName(names[column]);

            if (numbers.Contains(names[column]) && row[column] is { Length: > 0 } number)
            {
                writer.WriteRawValue(number);
            }
            else
            {
                writer.WriteStringValue(row[column]);
            }
        }

        writer.WriteEndObject();
    }

    writer.WriteEndArray();
}

Console.WriteLine(Encoding.UTF8.GetString(File.ReadAllBytes("titanic.json").AsSpan(0, 400)));
