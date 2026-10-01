// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A feature that held one value on every training row taught a network nothing about how its answer moves with it: where
/// that value was nought, no training row ever moved the first layer's weights for it, and they are still the random start's.
/// So the network records, with what it was trained on, each such feature and its value; it names, for each row served
/// later, the features the row moves away from that value — a category the training rows never held, a gap where they held
/// none — and the report counts, part by part, the rows that move one.
/// </summary>
public class UnfamiliarTests
{
    private const long Seed = 20260929;

    private static readonly string[] Columns = ["pclass", "sex", "age", "sibsp", "parch", "fare"];

    private static readonly string[] Reserved = ["pclass_other", "pclass_was_missing", "sex_other", "sex_was_missing"];

    [Fact]
    public void TheFeaturesEveryTrainingRowHeldAtNought_KeepTheirRandomStart_WhileEveryOtherFeaturesWeightsMove()
    {
        var trained = Network().Fit(Passengers(reported: false), new FitOptions(Seed) { Epochs = 5 });
        var start = new Sequential().Dense(16).Relu().Dense(1).Lower(new Shape(14), new RandomStream(Seed));
        var names = trained.TrainedOn.Features;

        for (var input = 0; input < names.Count; input++)
        {
            if (Reserved.Contains(names[input]))
            {
                Assert.Equal(FirstLayer(start, input), FirstLayer(trained.Network, input));
            }
            else
            {
                Assert.NotEqual(FirstLayer(start, input), FirstLayer(trained.Network, input));
            }
        }
    }

    [Fact]
    public void ANetwork_RecordsEachFeatureThatHeldOneValueOnEveryTrainingRow_WithThatValue()
    {
        var trained = Network().Fit(Passengers(reported: false), new FitOptions(Seed) { Epochs = 2 });

        Assert.Equal(Reserved, trained.TrainedOn.Unvaried!.Keys);
        Assert.All(trained.TrainedOn.Unvaried.Values, value => Assert.Equal(0, value));
    }

    [Fact]
    public void APassengerWrittenMale_OfClassFour_OrOfNoSex_IsNamedWithTheFeatureTheyMove_AndAFamiliarOneWithNone()
    {
        var trained = Network().Fit(Passengers(reported: false), new FitOptions(Seed) { Epochs = 2 });

        var predictions = trained.Predict(Strangers());

        Assert.Equal<IReadOnlyList<string>>([[], ["sex_other"], ["pclass_other"], ["sex_was_missing"], ["pclass_other", "sex_other"]], predictions.Unfamiliar!);
        Assert.Equal([0, 1, 2, 3, 4], predictions.HandedInAt);
        Assert.Equal(5, predictions.Answers.Count);
    }

    [Fact]
    public void ANetworkReadFromItsFile_NamesThemAlike_ForItsFileCarriesTheRecord()
    {
        var trained = Network().Fit(Passengers(reported: false), new FitOptions(Seed) { Epochs = 2 });
        var json = trained.ToJson();

        var loaded = TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
        var network = JsonNode.Parse(json)!["network"]!;
        var written = network["trainedOn"]!["unvaried"]!.AsObject();

        Assert.Equal(trained.TrainedOn.Unvaried!, loaded.TrainedOn.Unvaried!);
        Assert.Equal<IReadOnlyList<string>>(trained.Predict(Strangers()).Unfamiliar!, loaded.Predict(Strangers()).Unfamiliar!);
        Assert.Equal(json, loaded.ToJson());
        Assert.Equal(2, network["version"]!.GetValue<int>());
        Assert.Equal(Reserved, written.Select(member => member.Key));
        Assert.All(written, member => Assert.Equal(0, member.Value!.GetValue<double>()));
    }

    [Fact]
    public void ThePassengersReport_CountsNoRowOnAnyPart_ForNoRowItMeasuresMovesOne()
    {
        var trained = Network().Fit(Passengers(reported: true), new FitOptions(Seed) { Epochs = 2 });

        Assert.Equal([Part.Train, Part.Validation, Part.Test], trained.Measures!.Parts.Select(part => part.Part));
        Assert.Equal<int?>([0, 0, 0], trained.Measures.Parts.Select(part => part.UnfamiliarRows));
    }

    [Fact]
    public void AReport_CountsTheRowsOfEachPartThatMoveOne_AsServingNamesThem()
    {
        // Twenty rows t = 1…20 of x = sin t, whose answer is whether x is above nought: ten train, five judge, five test.
        // Every training row is red or blue; one row that judges is of no colour, and one that tests is green.
        var prepared = Pdd.Create()
            .Read(
                CsvRowSource.FromText("t,colour,x,y\n" + string.Join('\n', Enumerable.Range(1, 20).Select(t => string.Create(
                    CultureInfo.InvariantCulture,
                    $"{t},{(t == 13 ? string.Empty : t == 18 ? "green" : t % 2 == 0 ? "red" : "blue")},{Math.Sin(t)},{(Math.Sin(t) > 0 ? 1 : 0)}"))) + "\n"),
                "twenty rows")
            .Declare(schema => schema.Integer("t", "y").Category("colour").Number("x"))
            .SplitByTime("t", 0.50, 0.25)
            .EncodeCategories()
            .Normalise("x", Scale.MidRange)
            .Drop("t")
            .Target("y")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        var trained = new Sequential().Dense(4).Tanh().Dense(1).Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(prepared, new FitOptions(seed: 3) { Epochs = 2 });
        var served = trained.Predict(CsvRowSource.FromText("t,colour,x\n18,green,-0.75\n13,,0.42\n2,red,0.9\n"));

        Assert.Equal(["colour_other", "colour_was_missing"], trained.TrainedOn.Unvaried!.Keys);
        Assert.Equal<int?>([0, 1, 1], trained.Measures!.Parts.Select(part => part.UnfamiliarRows));
        Assert.Equal<IReadOnlyList<string>>([["colour_other"], ["colour_was_missing"], []], served.Unfamiliar!);
    }

    private static CompiledNetwork Network() => new Sequential().Dense(16).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());

    // The wiki's Titanic pipeline: fourteen features on one scale, whether the passenger survived, and — when reported — a
    // report measuring the network on every part.
    private static PreparedData Passengers(bool reported)
    {
        var declared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived");

        return (reported ? declared.Report(report => report.Measure(Metric.Accuracy).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers)) : declared)
            .Build()
            .Run();
    }

    // The README's passenger as the training rows wrote him; then written 'Male', of a fourth class, of no sex given, and
    // both of the first two at once.
    private static InMemoryRowSource Strangers() =>
        new(
            Columns,
            [
                ["3", "male", "22", "1", "0", "7.25"],
                ["3", "Male", "22", "1", "0", "7.25"],
                ["4", "male", "22", "1", "0", "7.25"],
                ["3", "", "22", "1", "0", "7.25"],
                ["4", "Male", "22", "1", "0", "7.25"],
            ]);

    // What the first layer weighs one input by, towards each of its sixteen units.
    private static float[] FirstLayer(Network network, int input) =>
        [.. network.Slots().Single(named => named.Path == "0.weight").Slot.Value.Values.ToArray().Skip(input * 16).Take(16)];
}
