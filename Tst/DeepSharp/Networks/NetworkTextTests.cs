// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network's file is parsed without its tensors' values, which are read from the text where they stand: the parsed text
/// holds every tensor's values as an empty list — a slot's under the parameters and the running statistics, what the
/// optimizer remembers, the best epoch's slots — and every other list as it is written.
/// </summary>
public class NetworkTextTests
{
    private static readonly string Checkpoint = """
        {
          "network": {
            "version": 2,
            "layers": {"kind": "stack", "layers": [{"kind": "batchNorm", "features": 2, "momentum": 0.1, "epsilon": 1e-05}]},
            "parameters": {"0.weight": {"shape": [2], "values": [1, 2]}, "0.bias": {"shape": [2], "values": [3, 4]}},
            "state": {"0.running_mean": {"shape": [2], "values": [5, 6]}, "0.running_var": {"shape": [2], "values": [7, 8]}},
            "loss": {"kind": "meanSquaredError"}
          },
          "training": {
            "version": 2,
            "seed": 5,
            "optimizer": {"kind": "adam", "rate": 0.1, "betas": [0.9, 0.999], "epsilon": 1e-08},
            "memory": {"0.weight": {"steps": 1, "tensors": {"exp_avg": {"shape": [2], "values": [9, 10]}}}},
            "judgement": {
              "bestSlots": {
                "parameters": {"0.weight": {"shape": [2], "values": [11, 12]}},
                "state": {"0.running_mean": {"shape": [2], "values": [13, 14]}}
              }
            },
            "history": [{"number": 0, "loss": 1, "learningRate": 0.1}]
          }
        }
        """.ReplaceLineEndings("\n");

    [Theory]
    [InlineData("network/parameters/0.weight")]
    [InlineData("network/parameters/0.bias")]
    [InlineData("network/state/0.running_mean")]
    [InlineData("network/state/0.running_var")]
    [InlineData("training/memory/0.weight/tensors/exp_avg")]
    [InlineData("training/judgement/bestSlots/parameters/0.weight")]
    [InlineData("training/judgement/bestSlots/state/0.running_mean")]
    public void ATensorsValues_AreParsedAsAnEmptyList_AndCountedAndReadWhereTheyStand(string tensor)
    {
        var text = new NetworkText(Checkpoint, PartReader.ReadsApart);
        using var parsed = text.Parsed();
        string[] values = [.. tensor.Split('/'), "values"];

        Assert.Equal(0, At(parsed.RootElement, values).GetArrayLength());
        Assert.Equal(2, At(parsed.RootElement, values[..^1]).GetProperty("shape")[0].GetInt32());
        Assert.Equal(2, text.CountOf(values));
        Assert.Equal(2, text.FloatsOf(values, 2, "not read")!.Length);
    }

    [Fact]
    public void EveryOtherList_IsParsedAsItIsWritten()
    {
        using var parsed = new NetworkText(Checkpoint, PartReader.ReadsApart).Parsed();
        var root = parsed.RootElement;

        Assert.Equal(1, At(root, ["network", "layers", "layers"]).GetArrayLength());
        Assert.Equal([0.9, 0.999], At(root, ["training", "optimizer", "betas"]).EnumerateArray().Select(beta => beta.GetDouble()));
        Assert.Equal(1, At(root, ["training", "history"]).GetArrayLength());
        Assert.Equal(1, At(root, ["network", "parameters", "0.weight", "shape"]).GetArrayLength());
    }

    [Fact]
    public void AFileWithNoListItsReaderReadsApart_IsParsedWhole()
    {
        using var parsed = new NetworkText(Checkpoint, _ => false).Parsed();

        Assert.Equal(2, At(parsed.RootElement, ["network", "parameters", "0.weight", "values"]).GetArrayLength());
    }

    private static JsonElement At(JsonElement root, IEnumerable<string> path) => path.Aggregate(root, (element, key) => element.GetProperty(key));
}
