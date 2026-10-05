// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using static System.FormattableString;

namespace DeepSharp.Sample.Pipelines;

/// <summary>
/// The walked pipelines, each started where a host hands pipelines out: the passenger list, asked first what each of its
/// columns holds and profiled for what should not be there, and then started again from the table's course; and the price
/// series, split along its dates. The passengers' rows are handed over and served, and their pipeline is saved as a file and
/// run again from it.
/// </summary>
public static class PipelineSample
{
    /// <summary>Runs the sample.</summary>
    /// <param name="pipelines">Where a pipeline is started, as a host hands it out.</param>
    /// <param name="data">The folder the datasets are in.</param>
    /// <param name="output">Where what happens is told.</param>
    public static void Run(IPipelineFactory pipelines, string data, TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(pipelines);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(output);

        var titanic = Path.Join(data, "titanic.csv");

        Proposed(pipelines, titanic, output);
        Profiled(pipelines, titanic, output);

        var passengers = Passengers(pipelines, titanic);
        Report("Titanic", passengers, output);

        Report("Titanic, from the table's course", StartedFromACourse(titanic, output), output);

        var prices = Prices(pipelines, Path.Join(data, "apple.csv"));
        Report("Apple", prices, output);

        HeldBack(prices, output);
        HandedOverAndServed(passengers, output);
        Saved(passengers, output);
    }

    // Before anything is declared, the file is asked what it holds: every cell read, a kind proposed for each column. It
    // is a proposal and nothing more — the schema is still a person's to write.
    private static void Proposed(IPipelineFactory pipelines, string path, TextWriter output)
    {
        output.WriteLine("=== what titanic.csv says each column holds, before anything is declared ===");

        foreach (var column in pipelines.Create().ReadCsv(path).ProposedKinds().Columns)
        {
            var offered = column.Offered is { } other ? $", {other.Word()} offered" : "";
            var values = column.Distinct is { } distinct ? Invariant($", {distinct} values") : "";
            var gaps = column.Gaps > 0 ? Invariant($", {column.Gaps} gaps") : "";

            output.WriteLine($"  {column.Name,-12} {column.Kind.Word()}{offered}{values}{gaps}");
        }

        output.WriteLine();
    }

    // Every column the file holds, declared as it proposes, and a profile of the training rows: what should not be there,
    // each with how it is answered — a step, the column left out, or a value the schema says stands for a gap.
    private static void Profiled(IPipelineFactory pipelines, string path, TextWriter output)
    {
        var everything = pipelines.Create()
            .ReadCsv(path)
            .Declare(schema => schema
                .Integer("survived", "pclass", "sibsp", "parch")
                .Number("fare")
                .Optional("age", ColumnKind.Number)
                .Category("sex", "embarked", "class", "who", "embark_town")
                .Optional("deck", ColumnKind.Category)
                .Boolean("adult_male", "alive", "alone"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .Profile()
            .Target("survived")
            .Build()
            .Run();
        var profile = everything.Evidence.Values.OfType<DataProfile>().Single();

        output.WriteLine(Invariant($"=== what should not be there, measured on the {profile.Rows} training rows ==="));

        foreach (var alert in profile.Alerts)
        {
            output.WriteLine($"  {alert.Column}: {alert.Says} Answered by {Answering(alert.Answer)}.");
        }

        output.WriteLine();
    }

    // How an alert is answered, in the words the notebook's profile shows.
    private static string Answering(AlertAnswer answer) => answer.Action switch
    {
        AlertAction.LeaveOut => "leaving it out",
        AlertAction.SayMissing => $"saying in the schema that {answer.Value} stands for a gap",
        _ => answer.Verb!,
    };

    // A table of people: gaps of two very different sizes, a category, and no time column at all.
    private static PreparedData Passengers(IPipelineFactory pipelines, string path) =>
        pipelines.Create()
            .ReadCsv(path)
            .Declare(schema => schema
                .Integer("survived", "pclass", "sibsp", "parch")
                .Number("fare")
                .Optional("age", ColumnKind.Number)
                // Said here, where the data is declared, because which columns stand for a group is a fact about
                // the data rather than a decision about the model. EncodeCategories then takes them by name.
                .Category("sex", "embarked"))
            .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
            // Where this pipeline's features land, said once: between minus one and one, which is what a scaling that
            // names no kind then writes. Form.Unit would put them between nothing and one instead.
            .DefaultFeatures(Form.Signed)
            // The profile said what should not come along: 'alive' hands a model the answer, and 'class' and
            // 'embark_town' say again what 'pclass' and 'embarked' say. None of them is declared, so none comes along.
            // Whether a fare of 0 means that nobody knew it is for whoever knows the data to say; here it stays a fare.
            // Seventy to learn from, fifteen to choose with; what is measured on is never written down, it is
            // what is left. So this says fifteen too — and nothing can ask for more rows than there are.
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            // ---- nothing above this line is allowed to learn from the data ----
            // The median of the ages is learned from the training rows, so the fill stands here. Nothing is worked out
            // from 'age' above the line; if something were, its gaps would travel into it, and the answer would be to
            // settle them where the features are worked out, with a value no row decided — which is what SettleGaps is
            // for, and which writes nothing into the fitted half because it learned nothing. This line would then go:
            // a settled column has no gap left, so the pipeline refuses a fill of it rather than run a line that cannot act.
            .FillMissing(fill => fill.Median("age"))
            .EncodeCategories()
            .Normalise("age", "fare", "family")
            .Target("survived")
            .Build()
            .Run();

    // A course, started from instead of written out from memory: every step of the table's course in the order the steps belong,
    // each waiting for what only a person knows. This program brings no network, so the course leaves that step out; said for the
    // passenger list, the rest is the pipeline the passengers' chain above makes, in the order the course teaches.
    private static PreparedData StartedFromACourse(string path, TextWriter output)
    {
        var catalog = StepCatalog.BuiltIn();
        var course = PipelineCourse.Table.Without("learn.network");

        output.WriteLine("=== the table's course, before anything is said ===");

        foreach (var waiting in course.Waiting(catalog))
        {
            output.WriteLine($"  {waiting}");
        }

        output.WriteLine();

        return Pdd.From(
            course
                .Say("read.csv", new JsonObject { ["path"] = path })
                .Say("declare", Json("""
                    {"columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "sibsp", "kind": "integer", "optional": false},
                                 {"name": "parch", "kind": "integer", "optional": false}, {"name": "fare", "kind": "number", "optional": false},
                                 {"name": "age", "kind": "number", "optional": true}]}
                    """))
                .Say("settle.gaps", new JsonObject { ["column"] = "fare" })
                .Say("feature.add", new JsonObject { ["column"] = "family", ["left"] = "sibsp", ["arithmetic"] = "plus", ["right"] = "parch" })
                .Say("scale.given", new JsonObject { ["column"] = "fare", ["lowest"] = 0, ["highest"] = 512 })
                .Say("split.stratified", new JsonObject { ["column"] = "survived" })
                .Say("target", new JsonObject { ["column"] = "survived" })
                .Say("drop.columns", new JsonObject { ["columns"] = new JsonArray("sibsp", "parch") })
                .Say("fill.missing", new JsonObject { ["column"] = "age" })
                .Say("normalise", new JsonObject { ["column"] = "age" }),
            catalog).Run();
    }

    private static JsonObject Json(string text) => JsonNode.Parse(text)!.AsObject();

    // A series in time: the split runs along the date, and the day of the week is written as a place on a circle so that
    // Monday and Sunday are neighbours.
    private static PreparedData Prices(IPipelineFactory pipelines, string path) =>
        pipelines.Create()
            .ReadCsv(path)
            .Declare(schema => schema
                .Timestamp("Date")
                .Number("AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume")
                .Category("direction"))
            // The rows before a row are whichever the file happened to put there, unless an order is declared: the
            // same prices, reversed, gave a five-day average of 98.352 where the right one is 99.74. So every step
            // that looks at earlier rows stands below the one that says which rows are earlier.
            .OrderBy("Date")
            .AddFeature("range", "AAPL.High", Arithmetic.Minus, "AAPL.Low")
            // Indicators are borrowed from MatPlotLibNet rather than written again, and they stand above the line
            // because they learn nothing: arithmetic over the rows that came before, looking only backwards.
            .Add(new AddIndicatorStep("rsi", Indicator.Rsi, ["AAPL.Close"], 14))
            .Add(new AddIndicatorStep("atr", Indicator.Atr, ["AAPL.High", "AAPL.Low", "AAPL.Close"], 14))
            .Add(new AddIndicatorStep("bb", Indicator.BollingerBands, ["AAPL.Close"], 20))
            // A month is not a quantity and Tuesday is not two of anything, so the pieces of a moment arrive as
            // categories and the encoder takes them from there. Where time wraps round, a circle says it better.
            .TimeParts("Date", TimePart.Season, TimePart.Quarter)
            .Cyclical("Date", Period.DayOfWeek, Form.SplitSign)
            // An indicator of period N says nothing about the first N rows, and filling that would invent
            // measurements nobody took. So with indicators on the data, 506 rows are 487 rows and 19 of not-yet.
            .DropWarmUp()
            // Ten per cent held out of everything, to run the trained network over the way tomorrow's data will
            // arrive: nothing is fitted on it and nothing is measured on it. Train, validation and test are then
            // the ninety that remain — here 65, 15 and the 10 that are left.
            .Predict(10)
            .SplitByTime("Date", train: 65, validation: 15)
            .Normalise(scale => scale
                .Robust("AAPL.Close", "AAPL.Volume", "range")   // prices and volumes: the spike is not the data
                .MinMax("rsi"))                                 // already between nought and a hundred
            .EncodeCategories()
            .Build()
            .Run();

    // The newest rows, kept out of everything: no scale was learned from them and no score was taken on them, so running
    // the trained network over these is the closest thing to running it tomorrow.
    private static void HeldBack(PreparedData prices, TextWriter output)
    {
        var dates = (Column<DateTime>)prices.Table["Date"];
        var later = Enumerable.Range(0, prices.Table.RowCount)
            .Where(row => prices.Parts[row] == Part.Predict)
            .ToArray();

        output.WriteLine("=== the rows held back to predict on ===");
        output.WriteLine(Invariant($"  {later.Length} rows, {dates[later[0]]:yyyy-MM-dd} to {dates[later[^1]]:yyyy-MM-dd}")
                         + " — the newest days in the file, and the split in time is what puts them there");
        output.WriteLine();
    }

    // What a learner is handed: rows of numbers, their names in a fixed order, and the answers apart from them. And one
    // passenger nobody has seen, prepared with the numbers the training rows produced.
    private static void HandedOverAndServed(PreparedData passengers, TextWriter output)
    {
        var train = passengers.Batch(Part.Train);
        var test = passengers.Batch(Part.Test);

        output.WriteLine("=== handover ===");
        output.WriteLine(Invariant($"  train     {train.RowCount} rows x {train.Width} numbers, {train.Labels!.Count} answers"));
        output.WriteLine(Invariant($"  test      {test.RowCount} rows x {test.Width} numbers"));
        output.WriteLine($"  in order  {string.Join(", ", train.FeatureNames)}");
        output.WriteLine($"  first row {string.Join(", ", train.Features[0].Select(number => Invariant($"{number:0.##}")))}");
        output.WriteLine();

        var arriving = new InMemoryRowSource(
            ["survived", "pclass", "sibsp", "parch", "sex", "embarked", "fare", "age"],
            [["0", "3", "0", "0", "female", "S", "7.75", null]]);
        var served = passengers.Replay(arriving);

        output.WriteLine("=== one row arriving later ===");
        output.WriteLine(Invariant($"  age was missing  {((Column<double>)served["age_was_missing"])[0]}"));
        output.WriteLine(Invariant($"  age now          {((Column<double>)served["age"])[0]:0.####} (scaled by what training learned)"));
        output.WriteLine();
    }

    // The whole pipeline, both halves, as it would be saved beside a model — and the same declaration, read back from that
    // file and run again with no builder and no host in sight.
    private static void Saved(PreparedData passengers, TextWriter output)
    {
        output.WriteLine("The Titanic pipeline, as a file:");
        output.WriteLine(passengers.ToJson());

        var again = new Pipeline(PipelineDeclaration.FromJson(passengers.ToJson(), StepCatalog.BuiltIn())).Run();

        output.WriteLine(Invariant($"Read back and run again: {again.Table.Columns.Count} columns, ")
                         + Invariant($"{again.CountIn(Part.Train)} training rows — identical: ")
                         + $"{again.Declaration.Equals(passengers.Declaration)}");
    }

    private static void Report(string what, PreparedData prepared, TextWriter output)
    {
        output.WriteLine($"=== {what} ===");
        output.WriteLine(Invariant($"  rows      {prepared.Table.RowCount}")
                         + Invariant($" (train {prepared.CountIn(Part.Train)},")
                         + Invariant($" validation {prepared.CountIn(Part.Validation)},")
                         + Invariant($" test {prepared.CountIn(Part.Test)},")
                         + Invariant($" predict {prepared.CountIn(Part.Predict)})"));
        output.WriteLine($"  columns   {string.Join(", ", prepared.Table.Columns.Select(column => column.Name))}");

        foreach (var (at, values) in prepared.Fitted.OrderBy(each => each.Key))
        {
            var step = prepared.Declaration.Steps[at].Verb;

            foreach (var (name, value) in values.Numbers)
            {
                output.WriteLine(Invariant($"  learned   {step}[{at}] {name} = {value:0.####}"));
            }

            foreach (var (name, list) in values.Lists)
            {
                output.WriteLine($"  learned   {step}[{at}] {name} = {string.Join(", ", list)}");
            }
        }

        output.WriteLine();
    }
}
