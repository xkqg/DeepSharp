// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A declared part as the form writes it into a step's JSON and draws it from there: its name under 'kind', then every
/// setting as the value it holds — and several parts as a list, one as itself.
/// </summary>
public sealed class PartDeclarationExtensionsTests
{
    private static readonly PartKind Relu = new("relu", "Leaves what is below nought at nought.", []);

    [Fact]
    public void APart_IsWrittenAsItsNameThenEverySettingAsTheValueItHolds_ANumberAWordAndWhetherItIsSo()
    {
        var part = new PartDeclaration("mixed", [
            new PartSetting("rate", PartValue.Of(0.5)),
            new PartSetting("named", PartValue.Of("words")),
            new PartSetting("best", PartValue.Of(true)),
        ]);

        Assert.Equal("""{"kind":"mixed","rate":0.5,"named":"words","best":true}""", part.AsJson().ToJsonString());
    }

    [Fact]
    public void SeveralParts_AreWrittenAsAList_AndOneAsItself_AsTheParameterWritesThem()
    {
        var layers = new PartsParameter("layers", "The layers.", [Relu.Declared()], [Relu]);
        var single = new PartsParameter("layer", "The layer.", [Relu.Declared()], [Relu]) { Single = true };

        Assert.Equal("""[{"kind":"relu"},{"kind":"relu"}]""", layers.AsJson([Relu.Declared(), Relu.Declared()]).ToJsonString());
        Assert.Equal("""{"kind":"relu"}""", single.AsJson([Relu.Declared()]).ToJsonString());
    }
}
