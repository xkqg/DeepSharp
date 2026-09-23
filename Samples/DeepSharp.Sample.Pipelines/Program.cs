// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

// A pipeline reached the way an application reaches everything else: a builder, its services, and the app
// that comes out of it. None of it is required -- the last lines do the same with no host anywhere -- but
// this is the shape a .NET application already has, so the library fits into it.
var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddDeepSharpPipelines();

using var app = builder.Build();

var pipelines = app.Services.GetRequiredService<IPipelineFactory>();
var data = Path.Join(AppContext.BaseDirectory, "..", "..", "..", "..", "data");

// ---------------------------------------------------------------------------------------------------
// A table of people: gaps of two very different sizes, a category, and no time column at all.
// ---------------------------------------------------------------------------------------------------
var passengers = pipelines.Create()
    .ReadCsv(Path.Join(data, "titanic.csv"))
    .Declare(schema => schema
        .Integer("survived", "pclass", "sibsp", "parch")
        .Number("fare")
        .Optional("age", ColumnKind.Number)
        // Said here, where the data is declared, because which columns stand for a group is a fact about
        // the data rather than a decision about the model. EncodeCategories then takes them by name.
        .Category("sex", "embarked"))
    .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
    // 'alive' and 'class' are 'survived' and 'pclass' written as words. Neither was declared, so neither
    // comes along: a pipeline that carried everything in the file would hand a model its own answer.
    .SplitStratified("survived", train: 0.70, validation: 0.15, test: 0.15)
    // ---- nothing above this line is allowed to learn from the data ----
    .FillMissing("age", With.Median)
    .EncodeCategories()
    .Normalise("age", "fare", "family")
    .Target("survived")
    .Build()
    .Run();

Report("Titanic", passengers);

// ---------------------------------------------------------------------------------------------------
// A series in time: the split runs along the date, and the day of the week is written as a place on a
// circle so that Monday and Sunday are neighbours.
// ---------------------------------------------------------------------------------------------------
var prices = pipelines.Create()
    .ReadCsv(Path.Join(data, "apple.csv"))
    .Declare(schema => schema
        .Timestamp("Date")
        .Number("AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume")
        .Category("direction"))
    .AddFeature("range", "AAPL.High", Arithmetic.Minus, "AAPL.Low")
    // Indicators are borrowed from MatPlotLibNet rather than written again, and they stand above the line
    // because they learn nothing: arithmetic over the rows that came before, looking only backwards.
    .AddIndicator("rsi", Indicator.Rsi, ["AAPL.Close"], 14)
    .AddIndicator("atr", Indicator.Atr, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 14)
    .AddIndicator("bb", Indicator.BollingerBands, ["AAPL.Close"], 20)
    // A month is not a quantity and Tuesday is not two of anything, so the pieces of a moment arrive as
    // categories and the encoder takes them from there. Where time wraps round, a circle says it better.
    .TimeParts("Date", TimePart.Season, TimePart.Quarter)
    .Cyclical("Date", Period.DayOfWeek, Form.SplitSign)
    // An indicator of period N says nothing about the first N rows, and filling that would invent
    // measurements nobody took. So with indicators on the data, 506 rows are 487 rows and 19 of not-yet.
    .DropWarmUp()
    .SplitByTime("Date", train: 0.70, validation: 0.15, test: 0.15)
    .Normalise("AAPL.Close", Scale.Robust)
    .Normalise("AAPL.Volume", Scale.Robust)
    .Normalise("range", Scale.Robust)
    .Normalise("rsi", Scale.MinMax)
    .EncodeCategories()
    .Build()
    .Run();

Report("Apple", prices);

// What a learner is handed: rows of numbers, their names in a fixed order, and the answers apart from them.
var train = passengers.Batch(Split.Train);
var test = passengers.Batch(Split.Test);

Console.WriteLine("=== handover ===");
Console.WriteLine($"  train     {train.RowCount} rows x {train.Width} numbers, {train.Labels!.Count} answers");
Console.WriteLine($"  test      {test.RowCount} rows x {test.Width} numbers");
Console.WriteLine($"  in order  {string.Join(", ", train.FeatureNames)}");
Console.WriteLine($"  first row {string.Join(", ", train.Features[0].Select(number => number.ToString("0.##")))}");
Console.WriteLine();

// Serving: one passenger nobody has seen, prepared with the numbers the training rows produced.
var arriving = new InMemoryRowSource(
    ["survived", "pclass", "sibsp", "parch", "sex", "embarked", "fare", "age"],
    [["0", "3", "0", "0", "female", "S", "7.75", null]]);

var served = passengers.Replay(arriving);

Console.WriteLine("=== one row arriving later ===");
Console.WriteLine($"  age was missing  {((Column<double>)served["age_was_missing"])[0]}");
Console.WriteLine($"  age now          {((Column<double>)served["age"])[0]:0.####} (scaled by what training learned)");
Console.WriteLine();

// The whole pipeline, both halves, as it would be saved beside a model.
Console.WriteLine("The Titanic pipeline, as a file:");
Console.WriteLine(passengers.ToJson());

// And the same declaration, read back from that file and run again with no builder in sight.
var again = new Pipeline(PipelineDeclaration.FromJson(passengers.ToJson())).Run();

Console.WriteLine($"Read back and run again: {again.Table.Columns.Count} columns, "
                  + $"{again.CountIn(Split.Train)} training rows — identical: "
                  + $"{again.Declaration.Equals(passengers.Declaration)}");

static void Report(string what, PreparedData prepared)
{
    Console.WriteLine($"=== {what} ===");
    Console.WriteLine($"  rows      {prepared.Table.RowCount}"
                      + $" (train {prepared.CountIn(Split.Train)},"
                      + $" validation {prepared.CountIn(Split.Validation)},"
                      + $" test {prepared.CountIn(Split.Test)})");
    Console.WriteLine($"  columns   {string.Join(", ", prepared.Table.Columns.Select(column => column.Name))}");

    foreach (var (at, values) in prepared.Fitted.OrderBy(each => each.Key))
    {
        var step = prepared.Declaration.Steps[at].Verb;

        foreach (var (name, value) in values.Numbers)
        {
            Console.WriteLine($"  learned   {step}[{at}] {name} = {value:0.####}");
        }

        foreach (var (name, list) in values.Lists)
        {
            Console.WriteLine($"  learned   {step}[{at}] {name} = {string.Join(", ", list)}");
        }
    }

    Console.WriteLine();
}
