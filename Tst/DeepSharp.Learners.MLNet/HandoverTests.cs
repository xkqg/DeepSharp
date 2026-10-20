// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Pipelines;
using Microsoft.ML;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// The rows a pipeline hands over, as ML.NET reads them.
/// </summary>
/// <remarks>
/// Four shapes reach this door: an answer that is one of two classes or a number, and rows that carry their answer or
/// are the question. Each is read here, because the one that is wrong silently is the one that trains on noughts.
/// </remarks>
public class HandoverTests
{
    [Fact]
    public void AnAnswerOfTwoClasses_RidesAsOneTrueOrFalse_AndANumberAsItself()
    {
        var context = new MLContext(1);
        var batch = new Batch(["a", "b"], [[1.0, 2.0], [3.0, 4.0]], Labels: [0, 1]);

        Assert.Equal(2, context.Data.CreateEnumerable<Scored>(HandedRows.Of(context, batch, classes: true), reuseRowObject: false).Count());
        Assert.Equal(2, context.Data.CreateEnumerable<Scored>(HandedRows.Of(context, batch, classes: false), reuseRowObject: false).Count());
    }

    [Fact]
    public void RowsThatAreTheQuestion_CarryNoAnswerAndAreTakenAllTheSame()
    {
        // What a prediction is asked of: the features alone. Nothing may refuse them for lacking the answer, and
        // nothing may read an answer that is not there.
        var context = new MLContext(1);
        var asked = new Batch(["a"], [[1.0], [2.0]], Labels: null);

        Assert.Equal(2, context.Data.CreateEnumerable<Scored>(HandedRows.Of(context, asked, classes: true, answers: false), reuseRowObject: false).Count());
        Assert.Equal(2, context.Data.CreateEnumerable<Scored>(HandedRows.Of(context, asked, classes: false, answers: false), reuseRowObject: false).Count());
    }

    [Fact]
    public void APipelineNamingMoreThanOneAnswer_IsRefusedAtTheDoor()
    {
        var two = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .AddFeature("alone", "survived", Arithmetic.Times, "survived")
            .SplitStratified("survived", 0.70, 0.15)
            .Labels(["survived", "alone"])
            .WithML(trainer => trainer.FastTree(trees: 5))
            .Build();

        var refused = Assert.Throws<InvalidOperationException>(two.TrainWithML);

        Assert.Contains("one answer", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RowsThatAreImages_AreHandedToATreeAsTheRowOfTheirPixels_AndAServedImageIsAnsweredAsATrainingOneIs()
    {
        // Images of 6 by 6, the answer being whether the left half is brighter than the right: a tree reads the pixels flattened,
        // with no shape stated, as it reads any other row of numbers.
        static InMemoryRowSource Images(int count, int offset)
        {
            var random = new Random(41 + offset);

            return new InMemoryRowSource(
                ["image", "bright"],
                [.. Enumerable.Range(0, count).Select(_ =>
                {
                    var leftBright = random.NextDouble() < 0.5;
                    var pixels = new byte[36];

                    for (var at = 0; at < pixels.Length; at++)
                    {
                        pixels[at] = (byte)((((at % 6) < 3) == leftBright ? 170 : 60) + random.Next(40));
                    }

                    return new string?[] { MediaCell.Pixels([6, 6, 1], pixels), leftBright ? "1" : "0" };
                })]);
        }

        var trained = Pdd.Create()
            .Read(Images(300, 0), "three hundred images")
            .Declare(schema => schema.Media("image").Integer("bright"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Target("bright")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers))
            .WithML(trainer => trainer.FastTree(trees: 20))
            .Build()
            .TrainWithML();
        var accuracy = trained.Measures!.Parts.Single().Values.Single(value => value.Metric == Metric.Accuracy);
        var answered = trained.Predict(Images(40, 5));
        var held = Images(40, 5).Rows.Select(row => row[1] == "1" ? 1 : 0).ToArray();

        Assert.True(accuracy.Value > 0.9, $"accuracy {accuracy.Value}");
        Assert.Equal(36, trained.TrainedOn.Features.Count);
        Assert.True(Enumerable.Range(0, 40).Count(at => (answered[at][0] >= 0.5 ? 1 : 0) == held[at]) >= 34);
    }

    /// <summary>One row as the view hands it on, read back to count what went in.</summary>
    private sealed class Scored
    {
        public float[] Features { get; init; } = [];
    }
}
