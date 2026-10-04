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

    /// <summary>One row as the view hands it on, read back to count what went in.</summary>
    private sealed class Scored
    {
        public float[] Features { get; init; } = [];
    }
}
