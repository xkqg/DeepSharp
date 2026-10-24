// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The rules that make steps one network: each takes what the step before it made unless it names others, a name is a path and
/// not a number, nothing is taken from a step that has not run, nothing is made that nothing uses, and a layer is run once. A stack
/// of layers alone is written to its file as it always was, and a stack that names its steps or takes another than the one before
/// is the eighth version of a network's part, which an older library refuses by its number.
/// </summary>
public class BranchStacksTests
{
    private static Dense Dense(int inputs, int outputs) => new(Tensor.Zeros(new Shape(inputs, outputs)), Tensor.Zeros(new Shape(outputs)));

    [Fact]
    public void AStackOfLayersAlone_DoesNotBranch_AndOneWhoseStepTakesAnotherThanTheStepBeforeIt_Does()
    {
        Assert.False(new LayerStack(Dense(3, 3), new Relu(), Dense(3, 1)).Branches);
        Assert.False(new LayerStack([new Step(Dense(3, 2), "hidden"), new Step(new Relu())]).Branches);
        Assert.True(new LayerStack([new Step(Dense(3, 3), "a"), new Step(new Sum(), Takes: ["input", "a"])]).Branches);
    }

    [Fact]
    public void TheShapesOfEachStep_AreThoseOfTheFirstStepItTakesAndOfWhatItMakes_AnExampleOfNothingSentThroughTheLayersThemselves()
    {
        var network = new LayerStack([new Step(Dense(6, 4), "a"), new Step(new Relu(), "ra"), new Step(new Concatenation(), Takes: ["input", "ra"]), new Step(Dense(10, 2))]);

        var shapes = network.Shapes(new Shape(6));

        Assert.Equal([new Shape(1, 6), new Shape(1, 4), new Shape(1, 6), new Shape(1, 10)], shapes.Select(each => each.Reaching));
        Assert.Equal([new Shape(1, 4), new Shape(1, 4), new Shape(1, 10), new Shape(1, 2)], shapes.Select(each => each.Made));
    }

    [Fact]
    public void AStackOfLayersAlone_IsWrittenAsItAlwaysWas_TheSecondVersionWithNothingBesideItsLayers()
    {
        var network = new LayerStack(Dense(3, 3), new Relu(), Dense(3, 1));
        var written = Written(network);

        Assert.Equal(2, VersionOf(written));
        Assert.DoesNotContain("names", written, StringComparison.Ordinal);
        Assert.DoesNotContain("takes", written, StringComparison.Ordinal);
        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias"], network.Slots().Select(slot => slot.Path));
        Assert.Equal(["0", "1", "2"], network.Steps.Select(step => step.Name));
        Assert.Equal([["input"], ["0"], ["1"]], network.Steps.Select(step => step.Takes.ToArray()));
    }

    [Fact]
    public void AStepThatIsNamed_IsTheEighthVersion_AndItsNameIsThePathOfWhatItsLayerHolds()
    {
        var network = new LayerStack([new Step(Dense(3, 2), "hidden"), new Step(new Relu()), new Step(Dense(2, 1), "out")]);
        var written = Written(network);

        Assert.Equal(8, VersionOf(written));
        Assert.Contains("\"names\"", written, StringComparison.Ordinal);
        Assert.DoesNotContain("takes", written, StringComparison.Ordinal);
        Assert.Equal(["hidden.weight", "hidden.bias", "out.weight", "out.bias"], network.Slots().Select(slot => slot.Path));
        Assert.Equal(["hidden", "1", "out"], network.Steps.Select(step => step.Name));
    }

    [Fact]
    public void AStepThatTakesAnotherThanTheOneBeforeIt_IsTheEighthVersion_AndIsWrittenAndReadBackAsItWas()
    {
        var network = new LayerStack([new Step(Dense(3, 3), "hidden"), new Step(new Relu(), "act"), new Step(new Sum(), "out", ["input", "act"])]);
        var written = Written(network);
        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());

        Assert.Equal(8, VersionOf(written));
        Assert.Contains("\"takes\"", written, StringComparison.Ordinal);
        Assert.Equal(written, Written(read.Network));
        Assert.Equal(network.Steps.Select(step => step.Takes), ((LayerStack)read.Network).Steps.Select(step => step.Takes));
    }

    [Fact]
    public void ANetworkPartOfTheEighthVersion_IsRefusedByAnOlderLibrary_ByItsNumber()
    {
        var written = Written(new LayerStack([new Step(Dense(2, 2), "a"), new Step(new Sum(), "out", ["input", "a"])]));
        var newer = written.Replace("\"version\": 8", "\"version\": 10", StringComparison.Ordinal);

        Assert.Contains("version 10", Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(newer, "network", NetworkCatalog.BuiltIn())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANamedStep_WhoseNameIsItsPlace_IsReadBackAsUnnamed()
    {
        var written = Written(new LayerStack([new Step(Dense(2, 2), "first"), new Step(new Relu())]));
        var read = (LayerStack)NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn()).Network;

        Assert.Equal(["first", "1"], read.Steps.Select(step => step.Name));
        Assert.Equal(written, Written(read));
    }

    [Fact]
    public void ANameThatCannotStandInAPath_OrIsAPlace_OrIsTheInput_IsRefused()
    {
        foreach (var name in new[] { "", " ", "a.b", "a,b", "3", "007", "input" })
        {
            Assert.Throws<ArgumentException>(() => new LayerStack([new Step(Dense(2, 2), name)]));
        }
    }

    [Fact]
    public void TwoStepsOfOneName_AreRefused()
    {
        Assert.Contains("'a'", Assert.Throws<ArgumentException>(() => new LayerStack([new Step(Dense(2, 2), "a"), new Step(new Relu(), "a")])).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepThatTakesWhatHasNotRun_OrWhatDoesNotExist_OrItself_IsRefused()
    {
        Assert.Contains("'later'", Assert.Throws<ArgumentException>(() => new LayerStack([new Step(new Sum(), "first", ["input", "later"]), new Step(new Relu(), "later")])).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(Dense(2, 2), "a"), new Step(new Sum(), "out", ["input", "nowhere"])]));
        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(new Relu(), "a", ["a"])]));
    }

    [Fact]
    public void AStepNothingTakes_IsRefused_ForItsWorkWouldBeMadeAndNeverUsed()
    {
        var refused = Assert.Throws<ArgumentException>(() => new LayerStack([new Step(Dense(2, 2), "unused"), new Step(new Relu(), "next", ["input"])]));

        Assert.Contains("'unused'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerThatWorksOnOneStep_IsRefusedTwo_AndAJoinOneOrNone()
    {
        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(Dense(2, 2), "a"), new Step(new Relu(), "out", ["input", "a"])]));
        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(new Sum(), "out", ["input"])]));
        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(new Concatenation(), "out", [])]));
    }

    [Fact]
    public void ALayerAtTwoSteps_IsRefused_ForALayerBelongsToOneNetworkAndIsRunOnce()
    {
        var shared = new Relu();

        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(shared, "a"), new Step(shared, "b")]));
    }

    [Fact]
    public void ANetworkWithNoSteps_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new LayerStack(Array.Empty<Step>()));
        Assert.Throws<ArgumentException>(() => new LayerStack(Array.Empty<Layer>()));
        Assert.Throws<ArgumentNullException>(() => new LayerStack((IEnumerable<Step>)null!));
        Assert.Throws<ArgumentNullException>(() => new LayerStack((IEnumerable<Layer>)null!));
    }

    [Fact]
    public void ASumOrAConcatenationHandedOneTensor_SaysItNeedsSeveral()
    {
        var pass = Pass.Evaluation(new CpuBackend());
        var rows = Tensor.Zeros(new Shape(2, 2));

        Assert.Contains("two steps", Assert.Throws<InvalidOperationException>(() => new Sum().Forward(rows, pass)).Message, StringComparison.Ordinal);
        Assert.Contains("two steps", Assert.Throws<InvalidOperationException>(() => new Concatenation().Forward(rows, pass)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StepsLaidSideBySide_MustBeAlikeInEveryAxisButTheLast_AndHaveAnAxisBesideTheRows()
    {
        var pass = Pass.Evaluation(new CpuBackend());
        var network = new LayerStack([new Step(new Columns(0, 1), "one"), new Step(new Concatenation(), "out", ["input", "one"])]);

        Assert.Equal(new Shape(3, 5), network.Forward(Tensor.Zeros(new Shape(3, 4)), pass).Shape);
        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(new Concatenation(), "out", ["input", "input"])]).Forward(Tensor.Zeros(new Shape(4)), pass));
        Assert.Throws<ArgumentException>(() => new LayerStack([new Step(new Flatten(), "flat"), new Step(new Concatenation(), "out", ["input", "flat"])]).Forward(Tensor.Zeros(new Shape(2, 3, 4)), pass));
    }

    [Fact]
    public void AColumnBlock_NeedsRowsThatReachIt_AndAStartAndACountThatMeanSomething()
    {
        var pass = Pass.Evaluation(new CpuBackend());

        Assert.Throws<ArgumentException>(() => new Columns(2, 3).Forward(Tensor.Zeros(new Shape(2, 4)), pass));
        Assert.Throws<ArgumentException>(() => new Columns(0, 1).Forward(Tensor.Zeros(new Shape(4)), pass));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Columns(-1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Columns(0, 0));
        Assert.Equal([2f, 3f], new Columns(1, 2).Forward(Tensor.From(new Shape(1, 4), [1f, 2f, 3f, 4f]), pass).Values.ToArray());
    }

    [Fact]
    public void AColumnBlockWithAnInvalidStartOrCount_IsRefusedWhereTheFileSaysIt()
    {
        var written = Written(new LayerStack([new Step(new Columns(0, 2), "block")]));

        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(written.Replace("\"count\": 2", "\"count\": 0", StringComparison.Ordinal), "network", NetworkCatalog.BuiltIn()));
        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(written.Replace("\"first\": 0", "\"first\": -1", StringComparison.Ordinal), "network", NetworkCatalog.BuiltIn()));
    }

    [Fact]
    public void AFileWhoseNamesOrTakesAreNotAsManyAsItsLayers_OrAreNotText_IsRefusedAtThem()
    {
        var written = Written(new LayerStack([new Step(Dense(2, 2), "a"), new Step(new Sum(), "out", ["input", "a"])]));
        var fewer = written.Replace("\"names\": [\n        \"a\",\n        \"out\"\n      ]", "\"names\": [\n        \"a\"\n      ]", StringComparison.Ordinal);
        var notText = written.Replace("\"names\": [\n        \"a\",\n        \"out\"\n      ]", "\"names\": [\n        \"a\",\n        3\n      ]", StringComparison.Ordinal);
        var takesFewer = written.Replace("[\n          \"input\",\n          \"a\"\n        ]", "\"input\"", StringComparison.Ordinal);

        Assert.NotEqual(written, fewer);
        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(fewer, "network", NetworkCatalog.BuiltIn()));
        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(notText, "network", NetworkCatalog.BuiltIn()));
        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(takesFewer, "network", NetworkCatalog.BuiltIn()));
    }

    [Fact]
    public void AFileWhoseStepsBreakTheRules_IsRefused_NamingTheRule()
    {
        var written = Written(new LayerStack([new Step(Dense(2, 2), "a"), new Step(new Sum(), "out", ["input", "a"])]));
        var later = written.Replace("\"input\",\n          \"a\"", "\"a\",\n          \"out\"", StringComparison.Ordinal);

        Assert.NotEqual(written, later);
        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(later, "network", NetworkCatalog.BuiltIn()));
    }

    [Fact]
    public void ANetworkWhoseLastStepIsASigmoidTheLossAppliesItself_IsRefusedAtCompile_AsAStackOfLayersIs()
    {
        var network = new LayerStack([new Step(Dense(2, 1), "a"), new Step(new Sigmoid(), "out")]);

        Assert.Contains("out", Assert.Throws<ArgumentException>(() => network.Compile(new Sgd(0.1), new BinaryCrossEntropy())).Message, StringComparison.Ordinal);
    }

    private static string Written(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(stream.ToArray()).Replace("\r\n", "\n", StringComparison.Ordinal);
    }

    private static int VersionOf(string written) => JsonDocument.Parse(written).RootElement.GetProperty("network").GetProperty("version").GetInt32();
}
