// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The graphs PyTorch's exporter makes of networks that read a series a step at a time (Fixtures/onnx-recurrent-fixtures.py made them):
/// an LSTM or a GRU node whose weights ONNX keeps outputs by inputs with its own gate order, its state started at nought by an Expand of a
/// constant, and the transposes, squeezes and reshapes that put what it made back as batch-first series, read into the layers of the same
/// names that answer as PyTorch answered.
/// </summary>
public class OnnxRecurrentFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(OnnxFixtures.Folder, "onnx-recurrent.json"))).RootElement;

    private static Tensor Examples(string name)
    {
        var written = Answers.GetProperty(name).GetProperty("examples");

        return Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), OnnxFixtures.Values(written.GetProperty("values")));
    }

    [Theory]
    [InlineData("lstm-last")]
    [InlineData("lstm-every")]
    [InlineData("gru-every")]
    [InlineData("gru-both")]
    public void TheGraphOfANetworkThatReadsASeriesAStepAtATime_AnswersAsPyTorchDid(string name)
    {
        using var graph = OnnxFixtures.Open($"onnx-recurrent-{name}.onnx");
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);
        var here = saved.Network.Predict(Examples(name), saved.Loss, Engine).Values.ToArray();
        var torch = OnnxFixtures.Values(Answers.GetProperty(name).GetProperty("outputs"));

        Assert.Equal(torch.Length, here.Length);
        Assert.InRange(torch.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Theory]
    [InlineData("lstm-last", typeof(Lstm), false, false)]
    [InlineData("lstm-every", typeof(Lstm), false, true)]
    [InlineData("gru-every", typeof(Gru), false, true)]
    [InlineData("gru-both", typeof(Gru), true, true)]
    public void TheNodeIsALayerOfItsOwnKind_WithTheDirectionsAndTheStepsItMakes(string name, Type kind, bool bothWays, bool sequences)
    {
        using var graph = OnnxFixtures.Open($"onnx-recurrent-{name}.onnx");
        var layers = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network).Layers;
        var recurrent = Assert.IsAssignableFrom<Recurrent>(layers[0]);

        Assert.IsType(kind, recurrent);
        Assert.Equal(new Recurrence(3, 4) { BothWays = bothWays, Sequences = sequences }, recurrent.Recurrence);
    }

    [Fact]
    public void TheGraphOfAStackOfTwoLayersInBothDirections_IsRefusedAtTheNodeThatTakesItsLastStatesOut()
    {
        using var graph = OnnxFixtures.Open("onnx-recurrent-lstm-two.onnx");
        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph));

        Assert.Contains("Gather", refused.Message, StringComparison.Ordinal);
    }

    private sealed record Edit(string File, Action<ModelProto> Change, string Said);

    private static void Output(ModelProto model, string name) => model.Graph.Output.Add(new ValueInfoProto { Name = name });

    private static void Constant(ModelProto model, string node, params long[] values) => model.Node(node).Attribute("value").T = OnnxFixtures.Wholes(node, values);

    private static readonly Dictionary<string, Edit> Refusals = new()
    {
        ["a node that reads from the end back"] = new("lstm-last", model => model.Node("/rnn/LSTM").SetText("direction", "reverse"), "it reads in the direction 'reverse'"),
        ["a node that takes its series batch first"] = new("lstm-last", model => model.Node("/rnn/LSTM").SetWhole("layout", 1), "layout 1"),
        ["a node that states its activations"] = new("lstm-last", model => model.Node("/rnn/LSTM").SetWholes("activation_alpha", 1), "activations, a clip or a coupled forget gate"),
        ["a node that has peepholes"] = new("lstm-last", model => model.Node("/rnn/LSTM").Input.Add("series"), "peepholes"),
        ["a node whose steps and last states are both taken"] = new("lstm-last", model => Output(model, "/rnn/LSTM_output_0"), "it gives its steps and its last states both"),
        ["a node whose last cell state is taken"] = new("lstm-last", model => Output(model, "/rnn/LSTM_output_2"), "its last cell state"),
        ["a node whose cell starts from a state the graph holds"] = new("lstm-last", model => model.Node("/rnn/LSTM").Input[6] = "series", "initial_c"),
        ["a node that resets before the product"] = new("gru-every", model => model.Node("/rnn/GRU").SetWhole("linear_before_reset", 0), "linear_before_reset 0"),
        ["a node whose state is another size than its weights"] = new("lstm-last", model => model.Node("/rnn/LSTM").SetWhole("hidden_size", 5), "recurrent weights R"),
        ["a node that is handed a series laid out batch first"] = new("lstm-last", model => model.Node("/rnn/LSTM").Input[0] = "series", "one step after another"),
        ["a squeeze of another axis than the direction's"] = new("lstm-every", model => Constant(model, "/rnn/Constant_5", 2), "drops axes [2]"),
        ["a gather of another state than the last"] = new("lstm-last", model => Constant(model, "/Constant", 1), "picks states out of those of a layer's directions"),
        ["a reshape of the steps into other widths"] = new("gru-both", model => Constant(model, "/rnn/Constant_3", 0, 0, 7), "[0, 0, 7]"),
        ["another operator on what a layer gave"] = new("lstm-every", model => model.Node("/rnn/Squeeze").OpType = "Neg", "works on a series laid out as ONNX's recurrent operators"),
        ["what a layer gave as the graph's output"] = new(
            "lstm-every",
            model =>
            {
                while (model.Graph.Node[^1].Name != "/rnn/Squeeze")
                {
                    model.Graph.Node.RemoveAt(model.Graph.Node.Count - 1);
                }

                model.Graph.Output[0].Name = "/rnn/Squeeze_output_0";
            },
            "with the steps before the batch"),
    };

    public static TheoryData<string> EveryRefusal => [.. Refusals.Keys];

    [Theory]
    [MemberData(nameof(EveryRefusal))]
    public void AGraphThatSaysWhatNoLayerHereDoes_IsRefusedAtTheNodeThatSaysIt(string name)
    {
        var refusal = Refusals[name];
        using var graph = OnnxFixtures.Edited($"onnx-recurrent-{refusal.File}.onnx", refusal.Change);

        Assert.Contains(refusal.Said, Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANodeThatLeavesItsBiasesOut_StartsFromBiasesOfNought_AsONNXDoes()
    {
        using var graph = OnnxFixtures.Edited("onnx-recurrent-lstm-last.onnx", model => model.Node("/rnn/LSTM").Input[3] = string.Empty);
        var layers = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network).Layers;
        var recurrent = Assert.IsAssignableFrom<Recurrent>(layers[0]);
        var biases = recurrent.Slots().Where(slot => slot.Path.StartsWith("bias", StringComparison.Ordinal)).ToList();

        Assert.Equal(["bias_ih_l0", "bias_hh_l0"], biases.Select(slot => slot.Path));
        Assert.All(biases, slot => Assert.All(slot.Slot.Value.Values.ToArray(), value => Assert.Equal(0f, value)));
    }

    [Fact]
    public void ARecurrentNodeGivenTheLengthsOfItsSeries_OrAStateTheGraphWorksOut_IsRefusedByName()
    {
        using var lengths = OnnxFixtures.Edited("onnx-recurrent-lstm-last.onnx", model => model.Node("/rnn/LSTM").Input[4] = "series");
        using var started = OnnxFixtures.Edited("onnx-recurrent-lstm-last.onnx", model => model.Node("/rnn/LSTM").Input[5] = "series");

        Assert.Contains("sequence_lens", Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(lengths)).Message, StringComparison.Ordinal);
        Assert.Contains("initial_h", Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(started)).Message, StringComparison.Ordinal);
    }
}
