// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network declared to look across the words of a review: the ids a pipeline cut it into, each its own vector at its place, attention over
/// them past the padding the pipeline declared, and the words of attention, positions and heads in the file with their settings.
/// </summary>
public class DeclaredAttentionTests
{
    private static readonly string[] Praise = ["great", "lovely", "fine", "superb"];
    private static readonly string[] Blame = ["awful", "dull", "bad", "boring"];
    private static readonly string[] Filler = ["film", "plot", "actor", "scene", "music", "ending"];

    private static InMemoryRowSource Reviews(int count)
    {
        var random = new Random(97);

        return new InMemoryRowSource(
            ["review", "good"],
            [.. Enumerable.Range(0, count).Select(_ =>
            {
                var good = random.NextDouble() < 0.5;
                var words = new List<string>();

                for (var at = random.Next(1, 3); at > 0; at--)
                {
                    words.Add((good ? Praise : Blame)[random.Next(4)]);
                }

                for (var at = random.Next(2, 6); at > 0; at--)
                {
                    words.Add(Filler[random.Next(6)]);
                }

                return new string?[] { string.Join(' ', words.OrderBy(_ => random.Next())), good ? "1" : "0" };
            })]);
    }

    private static PreparedData Prepared(Func<NetworkDeclaration, NetworkDeclaration> declared) =>
        Pdd.Create()
            .Read(Reviews(400), "four hundred reviews")
            .Declare(schema => schema.Text("review").Integer("good"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Tokenize("review", length: 8, vocabulary: 24)
            .Target("good")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .WithTorch(network => declared(network).Adam(0.02).BinaryCrossEntropy().Run(seed: 1, epochs: 40))
            .Build()
            .Run();

    private static NetworkDeclaration Attended(NetworkDeclaration steps) =>
        steps.TokenEmbedding(vocabulary: 24, width: 8, keepSteps: true).Named("steps")
            .Unpadded().Named("mask").Takes("input")
            .SinePositions().Named("placed").Takes("steps")
            .MultiHeadAttention(heads: 2).Named("attended").Takes("placed", "mask")
            .Flatten()
            .Dense(1);

    [Fact]
    public void ANetworkThatLooksAcrossTheWordsOfAReview_LearnsWhichWordsPraise_PastThePaddingTheMaskMarks()
    {
        var trained = Prepared(Attended).Train();
        var accuracy = trained.Measures!.Parts.Single(part => part.Part == Part.Test).Values.Single(value => value.Metric == Metric.Accuracy);
        var stack = Assert.IsType<LayerStack>(trained.Network);

        Assert.True(accuracy.Value > 0.8, $"accuracy {accuracy.Value}");
        Assert.Equal(["steps", "mask", "placed", "attended", "4", "5"], stack.Steps.Select(step => step.Name));
        Assert.Equal(["placed", "mask"], stack.Steps[3].Takes);
        Assert.Contains(stack.Slots(), slot => slot.Path == "attended.in_proj_weight");
        Assert.Contains(stack.Slots(), slot => slot.Path == "attended.out_proj.bias");
    }

    [Fact]
    public void TheDeclaredWords_AreInTheFileWithTheirSettings_AndOnesLeftOutMeanTheirDefaults()
    {
        var prepared = Prepared(steps => steps.TokenEmbedding(vocabulary: 24, width: 8, keepSteps: true).Named("steps")
            .SinePositions().Named("placed")
            .MultiHeadAttention(heads: 4, causal: true, dropout: 0.1).Named("looked")
            .Attention(heads: 2).Dense(1));
        var declared = Assert.IsType<LearnNetworkStep>(prepared.Declaration.Learner);
        var json = prepared.Declaration.ToJson();

        Assert.Equal(["tokenEmbedding", "sinePositions", "multiHeadAttention", "attention", "dense"], declared.Layers.Select(layer => layer.Kind));
        Assert.Equal(4, declared.Layers[2].Whole("heads"));
        Assert.True(declared.Layers[2].YesOrNo("causal"));
        Assert.Equal(0.1, declared.Layers[2].Number("dropout"));
        Assert.Equal(2, declared.Layers[3].Whole("heads"));
        Assert.False(declared.Layers[3].YesOrNo("causal"));
        Assert.Equal(0, declared.Layers[3].Number("dropout"));
        Assert.Equal(2, json.Split("\"causal\"").Length);
        Assert.Equal(prepared.Declaration, PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn().WithNetworks()));
    }
}
