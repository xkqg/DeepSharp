// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A prediction belongs to a row, and the row to a flock the schema numbers: the ids that travel with the rows into a
/// network travel out beside its answers, so nobody has to pair them up by position.
/// </summary>
public class PredictionIdsTests
{
    private static InMemoryRowSource Flocks(int count, bool reversed = false)
    {
        var rows = Enumerable.Range(1, count).Select<int, IReadOnlyList<string?>>(row =>
        [
            (1000 + row).ToString(CultureInfo.InvariantCulture),
            ((1000 + row) / 10.0).ToString(CultureInfo.InvariantCulture),
            (row * 2.5).ToString(CultureInfo.InvariantCulture),
        ]);

        return new InMemoryRowSource(["Flock", "Age", "Weight"], reversed ? rows.Reverse() : rows);
    }

    // The flock number is the id, or it is a column the pipeline leaves out: the one is carried, the other is nowhere.
    private static TrainedNetwork Trained(bool withId)
    {
        var declared = Pdd.Create().Read(Flocks(60), "flocks")
            .Declare(schema => (withId ? schema.Id("Flock", ColumnKind.Integer) : schema.Integer("Flock")).Number("Age", "Weight"));

        return (withId ? declared.SplitAtRandom(0.6, 0.2) : declared.Drop("Flock").SplitAtRandom(0.6, 0.2))
            .Target("Weight")
            .Normalise("Age", Scale.MidRange)
            .WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().Run(seed: 1, epochs: 2))
            .Build()
            .Train();
    }

    [Fact]
    public void ThePredictions_NameTheIdOfEveryRow_InTheOrderTheyAnswer()
    {
        var predictions = Trained(withId: true).Predict(Flocks(5, reversed: true));

        Assert.Equal(5, predictions.Answers.Count);
        Assert.Equal(predictions.HandedInAt.Select(at => (1005 - at).ToString(CultureInfo.InvariantCulture)), predictions.Ids);
    }

    [Fact]
    public void WithNoIdInTheSchema_ThePredictionsNameNone()
    {
        Assert.Null(Trained(withId: false).Predict(Flocks(5)).Ids);
    }

    [Fact]
    public void TheIds_AreNoPartOfWhatTheNetworkWasTrainedOn()
    {
        var trained = Trained(withId: true);

        Assert.Equal(["Age"], trained.TrainedOn.Features);
    }
}
