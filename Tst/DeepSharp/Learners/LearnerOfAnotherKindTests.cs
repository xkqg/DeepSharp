// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// One declaration, two learners of two kinds: a network learns from the run of every step, a tree that takes categories
/// itself from the run made for it. Both are measured by the pipeline's report on the same rows, their files share every
/// entry both learned, the tree's names the steps its run left out, and each learner is handed only its own kind of run. A
/// scale is nothing to the tree: it learns the same split with the scales left out as with them.
/// </summary>
public class LearnerOfAnotherKindTests
{
    private static readonly Part[] Parts = [Part.Train, Part.Validation, Part.Test];

    private static Pipeline Titanic => WikiTitanic.In(WikiTitanic.DataFolder);

    private static CompiledNetwork Small() => new Sequential().Dense(4).Tanh().Dense(1).Compile(new Adam(0.05), new BinaryCrossEntropy());

    [Fact]
    public void ANetworkAndATree_EachLearnFromTheRunMadeForIt_AndTheReportMeasuresBothOnTheSameRows()
    {
        var network = Small().Fit(Titanic.Run(), new FitOptions(seed: 20260929) { Epochs = 2 });
        var places = Titanic.RunFor(Stump.Needs);
        var tree = Stump.Fit(places.Batch(Part.Train, Stump.Needs));

        var measured = places.Measure([.. Parts.Select(part => tree.Predict(places.Batch(part, Stump.Needs)))]);

        Assert.Equal([623, 133, 135], network.Measures!.Parts.Select(part => part.Rows));
        Assert.Equal([623, 133, 135], measured.Parts.Select(part => part.Rows));
        Assert.Equal(KeysOf(network.Measures), KeysOf(measured));
        Assert.Equal("sex", tree.SplitOn);
        Assert.Equal(network.Measures.Metrics, measured.Metrics);
    }

    [Fact]
    public void TheTreesFile_NamesTheFiveStepsItsRunLeftOut_TheNetworksNone_AndWhatBothLearnedIsOneEntry()
    {
        var network = Small().Fit(Titanic.Run(), new FitOptions(seed: 20260929) { Epochs = 1 });
        using var networks = JsonDocument.Parse(network.ToJson());
        using var trees = JsonDocument.Parse(Titanic.RunFor(Stump.Needs).ToJson());
        var pipeline = networks.RootElement.GetProperty("pipeline");

        Assert.False(pipeline.TryGetProperty("skipped", out _));
        Assert.Equal(5, trees.RootElement.GetProperty("skipped").GetArrayLength());
        Assert.Equal(
            pipeline.GetProperty("fitted").EnumerateArray().Take(3).Select(entry => entry.GetRawText()),
            trees.RootElement.GetProperty("fitted").EnumerateArray().Select(entry => entry.GetRawText()));
    }

    [Fact]
    public void ThePassengerOnLineSeven_IsHandedToTheNetworkAsFourteenNumbers_AndToTheTreeAsNine_ClassAndSexAsTheirPlaces()
    {
        var every = Titanic.Run().Served(WikiTitanic.Line(7), Needs.OneScale);
        var places = Titanic.RunFor(Stump.Needs).Served(WikiTitanic.Line(7), Stump.Needs);

        Assert.Equal(
            ["sibsp", "parch", "age", "fare", "age_was_missing", "pclass_1", "pclass_2", "pclass_3", "pclass_other", "pclass_was_missing", "sex_female", "sex_male", "sex_other", "sex_was_missing"],
            every.FeatureNames);
        Assert.Equal([-1, -1, -0.281729, -0.966981, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], every.Features[0].Select(value => Math.Round(value, 6)));
        Assert.Equal(["sibsp", "parch", "age", "fare", "age_was_missing", "pclass", "pclass_was_missing", "sex", "sex_was_missing"], places.FeatureNames);
        Assert.Equal([0, 0, 29, 8.4583, 1, 2, 0, 1, 0], places.Features[0]);
        Assert.Equal(["1", "2", "3"], places.Categories!["pclass"]);
        Assert.Equal(["female", "male"], places.Categories["sex"]);
    }

    [Fact]
    public void ReadBackFromItsFile_TheTreesPipelineServesAPassengerAsNineNumbers_WithClassAndSexMarked()
    {
        var read = PreparedData.FromJson(Titanic.RunFor(Stump.Needs).ToJson(), StepCatalog.BuiltIn());
        var passenger = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"]]);

        var served = read.Served(passenger, Stump.Needs);

        Assert.Equal([1, 0, 22, 7.25, 0, 2, 0, 1, 0], served.Features[0]);
        Assert.Equal(["pclass", "sex"], served.Categories!.Keys.Order());
        Assert.Equal(["1", "2", "3"], served.Categories["pclass"]);
        Assert.Equal(["female", "male"], served.Categories["sex"]);
    }

    [Fact]
    public void AClassNoTrainingRowHeld_IsNamedUnfamiliarByTheTree_AndCountedByTheReport()
    {
        // The list with the passenger on line 18, in the test part, travelling in a fourth class.
        var folder = Directory.CreateTempSubdirectory("deepsharp-a8-").FullName;
        var lines = File.ReadAllLines(Repository.Data("titanic.csv"));
        var cells = lines[17].Split(',');
        cells[1] = "4";
        lines[17] = string.Join(',', cells);
        File.WriteAllLines(Path.Join(folder, "titanic.csv"), lines);

        var places = WikiTitanic.In(folder).RunFor(Stump.Needs);
        var tree = Stump.Fit(places.Batch(Part.Train, Stump.Needs));
        var test = places.Batch(Part.Test, Stump.Needs);
        var fourth = Enumerable.Range(0, places.Table.RowCount).Single(row => places.Table.Identities[row].ReadAt == 16);

        var measured = places.Measure([.. Parts.Select(part => tree.Predict(places.Batch(part, Stump.Needs)))]);

        Assert.Equal(Part.Test, places.Parts[fourth]);
        Assert.Equal(["1", "2", "3"], test.Categories!["pclass"]);
        Assert.Equal(["pclass"], tree.Unfamiliar(test.Features[test.Keys!.ToList().IndexOf(places.Table.Identities[fourth].Key)]));
        Assert.Equal<int?>([0, 0, 1], measured.Parts.Select(part => part.UnfamiliarRows));
    }

    [Fact]
    public void TheNetwork_IsRefusedTheTreesRun()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Small().Fit(Titanic.RunFor(Stump.Needs), new FitOptions(seed: 3) { Epochs = 1 }));

        Assert.Contains("step 5, 'encode.categories'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheTree_LearnsTheSameFromARunWithoutItsScales_AsFromARunOfEveryStep_RowForRow()
    {
        // A scale moves no row to the other side of a split between two training values, nor does holding one at the edge.
        var every = Titanic.Run();
        var noScale = Titanic.RunFor(Needs.NoScale);

        var fromEvery = Stump.Fit(every.Batch(Part.Train, Stump.Needs));
        var fromNoScale = Stump.Fit(noScale.Batch(Part.Train, Stump.Needs));

        foreach (var part in Parts)
        {
            Assert.Equal(
                every.Batch(part, Stump.Needs).Features.Select(fromEvery.Answer),
                noScale.Batch(part, Stump.Needs).Features.Select(fromNoScale.Answer));
        }

        Assert.Equal([5, 6, 7, 8], noScale.Skipped);
    }

    // The keys of the rows of every part measured, as the predictions' text carries them.
    private static string[] KeysOf(Measures measures)
    {
        using var text = JsonDocument.Parse(measures.PredictionsToJson());

        return [.. text.RootElement.GetProperty("parts").EnumerateArray().SelectMany(part => part.GetProperty("keys").EnumerateArray().Select(key => key.GetString()!))];
    }
}
