// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Charts;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Sample.Networks;

/// <summary>
/// The three walked networks, each behind the pipeline that prepares its rows: whether a passenger survived, a price five
/// days on, and a day's bikes hour by hour. Each is trained on its training rows and judged by its validation rows, measured
/// as its pipeline's report declares, saved as its one file, read back, and asked about a row served afresh.
/// </summary>
public static class NetworkSample
{
    private static readonly string[] Hours = [.. Enumerable.Range(0, 24).Select(hour => string.Create(CultureInfo.InvariantCulture, $"h{hour:00}"))];

    /// <summary>Runs the sample.</summary>
    /// <param name="data">The folder the three datasets are in.</param>
    /// <param name="output">Where what happens is told.</param>
    /// <param name="charts">The folder the charts are written into, as SVG.</param>
    public static void Run(string data, TextWriter output, string charts)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(charts);

        Passengers(data, output, charts);
        Prices(data, output, charts);
        Bikes(data, output, charts);
    }

    // Whether a passenger survived: Keras's words, every width worked out from the rows, the start drawn from the run's seed.
    private static void Passengers(string data, TextWriter output, string charts)
    {
        var prepared = Pdd.Create()
            .ReadCsv(Path.Join(data, "titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            // ---- nothing above this line learns from the rows ----
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            // Declared with the pipeline, before a number exists: every run is measured the same way.
            .Report(report => report
                .Measure(Metric.Accuracy, Metric.Precision, Metric.Recall, Metric.ConfusionMatrix)
                .On(Part.Train, Part.Validation, Part.Test)
                .As(Shown.Numbers, Shown.Drawn))
            .Build()
            .Run();

        var trained = new Sequential().Dense(16).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 100, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true } });

        var served = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]);
        var predictions = Told(output, "Titanic", trained, served);

        output.WriteLine(Invariant($"  a third-class man of 22: chance of surviving {predictions.Answers[0][0]:0.00}"));
        output.WriteLine(Invariant($"  a first-class woman of 38: chance of surviving {predictions.Answers[1][0]:0.00}"));

        Drawn(trained, Path.Join(charts, "titanic"));
        File.WriteAllText(Path.Join(charts, "titanic-confusion.svg"), trained.Measures!.ConfusionMatrices());
    }

    // A price five days on, answered as a return and brought back as a price: a network written as code, each layer drawing
    // its start from a stream of a person's own choosing.
    private static void Prices(string data, TextWriter output, string charts)
    {
        var path = Path.Join(data, "apple.csv");
        var prepared = Pdd.Create()
            .ReadCsv(path)
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close", "AAPL.Volume"))
            .OrderBy("Date")
            .Cyclical("Date", Period.DayOfWeek)
            .SplitByTime("Date", train: 0.70, validation: 0.15, gap: 5)
            .Ahead("AAPL.Close", 5, AheadAs.Return)
            // ---- nothing above this line learns from the rows ----
            .Normalise("AAPL.Close", Scale.MidRange)
            .Normalise("AAPL.Volume", Scale.MidRange)
            .Drop("Date")
            .Report(report => report.Measure(Metric.Rmse, Metric.Mae, Metric.R2).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers, Shown.Drawn))
            .Build()
            .Run();

        var stream = new RandomStream(20260929);
        var network = new LayerStack(new Dense(4, 3, stream.Draw("initialise:0", 0, 0)), new Tanh(), new Dense(3, 1, stream.Draw("initialise:2", 0, 0)));
        var trained = network.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 100, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true } });

        var lines = File.ReadAllLines(path);
        var predictions = Told(output, "Apple", trained, CsvRowSource.FromText($"{lines[0]}\n{lines[^1]}\n"));

        output.WriteLine(Invariant($"  the close five days on from {lines[^1].Split(',')[0]}: {predictions.Answers[0][0]:0.00}"));
        Drawn(trained, Path.Join(charts, "apple"));
    }

    // A day's bikes hour by hour: the shares of the day a network answers, brought back as bikes by the day's total.
    private static void Bikes(string data, TextWriter output, string charts)
    {
        var path = Path.Join(data, "bikes.csv");
        var prepared = Pdd.Create()
            .ReadCsv(path)
            .Declare(schema => schema
                .Timestamp("dteday")
                .Category("season", "weathersit")
                .Boolean("holiday", "workingday")
                .Number("temp", "atemp", "hum", "windspeed", "cnt")
                .Number(Hours))
            .Cyclical("dteday", Period.DayOfWeek)
            .Cyclical("dteday", Period.MonthOfYear)
            .SplitByTime("dteday", train: 0.70, validation: 0.15)
            .NormaliseRow(Norm.L1, Hours)
            // ---- nothing above this line learns from the rows ----
            .EncodeCategories()
            .Normalise("temp", Scale.MidRange)
            .Normalise("atemp", Scale.MidRange)
            .Normalise("hum", Scale.MidRange)
            .Normalise("windspeed", Scale.MidRange)
            .Drop("dteday")
            .Distribution(Hours, scaleBy: "cnt")
            .Drop("cnt")
            .Report(report => report.Measure(Metric.Rmse, Metric.Mae, Metric.R2).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers, Shown.Drawn))
            .Build()
            .Run();

        var trained = new Sequential().Dense(24)
            .Compile(new Adam(0.01), new CrossEntropy())
            .Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 100, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true } });

        var lines = File.ReadAllLines(path);
        var predictions = Told(output, "Bike Sharing", trained, CsvRowSource.FromText($"{lines[0]}\n{lines[^1]}\n"));
        var bikes = predictions.Answers[0];

        output.WriteLine(Invariant($"  bikes that day, {lines[^1].Split(',')[0]}: {bikes.Sum():0} in all, the busiest hour {Array.IndexOf(bikes, bikes.Max()):00}:00 with {bikes.Max():0}"));
        Drawn(trained, Path.Join(charts, "bikes"));
    }

    // What a run did, how its report measured it, and that its one file reads back to the same network: what it predicts
    // for the served rows, from the file as from the run.
    private static Predictions Told(TextWriter output, string name, TrainedNetwork trained, IRowSource served)
    {
        var history = trained.History!;

        output.WriteLine($"== {name}");
        output.WriteLine(Invariant(
            $"  {history.Epochs.Count} epochs ({(history.Stopped == Stopping.NoLongerImproving ? "stopped once no longer improving" : "every one given")}), the network kept from epoch {trained.TrainedOn.Epoch + 1}, seed {history.Seed}"));

        foreach (var part in trained.Measures!.Parts)
        {
            output.WriteLine($"  {part.Part.Word(),-10} " + string.Join("  ", part.Values.Select(measured => Invariant($"{measured.Metric.Word()} {measured.Value:0.###} (average {measured.Baseline:0.###})"))));
        }

        var file = trained.ToJson();
        var loaded = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
        var predictions = trained.Predict(served);
        var same = predictions.Answers.Zip(loaded.Predict(served).Answers).All(pair => pair.First.SequenceEqual(pair.Second));

        output.WriteLine(Invariant($"  saved as one file of {file.Length / 1024} KB; read back from its file, it predicts {(same ? "the same" : "otherwise")}"));

        return predictions;
    }

    // The loss curve of the run and the bars of its measures, each beside the training rows' average.
    private static void Drawn(TrainedNetwork trained, string stem)
    {
        File.WriteAllText($"{stem}-loss.svg", trained.History!.LossCurve());
        File.WriteAllText($"{stem}-measures.svg", trained.Measures!.Bars());
    }

    private static string Invariant(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);
}
