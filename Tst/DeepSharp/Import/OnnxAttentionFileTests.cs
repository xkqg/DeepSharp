// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// ONNX graphs that look across the steps of a series with the Attention operator of the twenty-third operator set (Fixtures/onnx-attention-fixtures.py
/// made them): the series projected by products with the weights the graph holds, attention over the heads, the answer projected and flattened. A
/// product of a series by weights is a dense layer over the channels of every step, and Attention is the layer of that name taking the
/// steps that made its queries, its keys and its values.
/// </summary>
public class OnnxAttentionFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(OnnxFixtures.Folder, "onnx-attention.json"))).RootElement;

    private static Tensor Examples(string name)
    {
        var written = Answers.GetProperty(name).GetProperty("examples");

        return Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), OnnxFixtures.Values(written.GetProperty("values")));
    }

    [Theory]
    [InlineData("attention")]
    [InlineData("attention-causal")]
    [InlineData("attention-self")]
    public void AGraphThatLooksAcrossASeries_AnswersAsOnnxRuntimeDid(string name)
    {
        using var graph = OnnxFixtures.Open($"onnx-{name}.onnx");
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);
        var here = saved.Network.Predict(Examples(name), saved.Loss, Engine).Values.ToArray();
        var ort = OnnxFixtures.Values(Answers.GetProperty(name).GetProperty("outputs"));

        Assert.Equal(ort.Length, here.Length);
        Assert.InRange(ort.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Theory]
    [InlineData("attention", false, 3)]
    [InlineData("attention-causal", true, 3)]
    [InlineData("attention-self", false, 1)]
    public void TheOperator_IsTheLayerAttention_TakingTheStepsThatMadeItsQueriesKeysAndValues(string name, bool causal, int takes)
    {
        using var graph = OnnxFixtures.Open($"onnx-{name}.onnx");
        var stack = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network);
        var attention = stack.Steps.Single(step => step.Layer is Attention);

        Assert.Equal(new Attending(2) { Causal = causal }, ((Attention)attention.Layer).Attending);
        Assert.Equal(takes, attention.Takes.Count);
    }

    private sealed record Edit(string File, Action<ModelProto> Change, string Said);

    private static readonly Dictionary<string, Edit> Refusals = new()
    {
        ["a mask handed to the operator"] = new("attention", model => model.Node("attention").Input.Add("query"), "attn_mask"),
        ["a scale of its own"] = new("attention", model => model.Node("attention").SetNumber("scale", 0.5f), "scale"),
        ["a softcap"] = new("attention", model => model.Node("attention").SetNumber("softcap", 2f), "softcap"),
        ["scores handed on"] = new("attention", model => model.Node("attention").SetWhole("qk_matmul_output_mode", 1), "qk_matmul_output_mode"),
        ["fewer heads of keys than of queries"] = new("attention", model => model.Node("attention").SetWhole("kv_num_heads", 1), "kv_num_heads"),
        ["no number of heads stated"] = new("attention", model => model.Node("attention").Unset("q_num_heads"), "q_num_heads"),
        ["a past of keys"] = new("attention", model => model.Node("attention").Input.AddRange(["", "query", "key"]), "past_key"),
    };

    public static TheoryData<string> EveryRefusal => [.. Refusals.Keys];

    [Theory]
    [MemberData(nameof(EveryRefusal))]
    public void AnAttentionThatSaysWhatNoLayerHereDoes_IsRefusedAtTheNodeThatSaysIt(string name)
    {
        var refusal = Refusals[name];
        using var graph = OnnxFixtures.Edited($"onnx-{refusal.File}.onnx", refusal.Change);

        Assert.Contains(refusal.Said, Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph)).Message, StringComparison.Ordinal);
    }
}
