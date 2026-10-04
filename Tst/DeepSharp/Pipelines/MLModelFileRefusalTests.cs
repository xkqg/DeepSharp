// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Every key a model file must say, and what it says when one is missing or is the wrong kind of thing.
/// </summary>
/// <remarks>
/// A model file is read by a library that cannot open the model inside it, so what it says about itself is all a reader
/// has. Each key is refused by name rather than read as a default, because a default here is a model answering from
/// something nobody wrote.
/// </remarks>
public class MLModelFileRefusalTests
{
    [Theory]
    [InlineData("rows", """{"version": 1, "mlnet": "5.0.0", "processor": "X64", "trainer": {"kind": "fastTree"}, "seed": 1, "answer": "a", "model": "", "pipeline": {"version": 7, "declaration": []}}""")]
    [InlineData("mlnet", """{"version": 1, "processor": "X64", "trainer": {"kind": "fastTree"}, "seed": 1, "answer": "a", "rows": 1, "model": "", "pipeline": {"version": 7, "declaration": []}}""")]
    [InlineData("answer", """{"version": 1, "mlnet": "5.0.0", "processor": "X64", "trainer": {"kind": "fastTree"}, "seed": 1, "rows": 1, "model": "", "pipeline": {"version": 7, "declaration": []}}""")]
    public void AKeyThatIsMissing_IsRefusedByItsOwnName(string key, string json)
    {
        var refused = Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson(json));

        Assert.Contains($"'{key}'", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATrainerThatSaysNoKind_ReadsAsNoTrainerAtAll()
    {
        // The settings are read whatever they are, and a trainer without a kind is a trainer nothing can be made of:
        // the file is read, and whoever would train from it finds no word to act on.
        var read = MLModelFile.FromJson(
            """
            {"version": 1, "mlnet": "5.0.0", "processor": "X64", "trainer": {"trees": 30}, "seed": 1,
             "answer": "a", "classes": false, "rows": 1, "model": "", "pipeline": {"version": 7, "declaration": []}}
            """);

        Assert.Equal(string.Empty, read.Trainer.Kind);
        Assert.Equal(30, Assert.Single(read.Trainer.Settings).Value.Number);
    }
}
