// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A step made under a verb from what is said and what is carried: the verb's own template, every value the step it
/// replaces holds under a key the verb takes, and on top of both every value said — so what is said wins, and what the
/// verb does not take is not carried. The step is read as a file's is, so the verb's own rules hold.
/// </summary>
public class StepMakingTests
{
    private static readonly StepCatalog Catalog = StepCatalog.BuiltIn();

    private static readonly string[] Bands = ["w500", "w550", "w600"];

    [Fact]
    public void AStepMade_IsItsTemplate_WithWhatItCarries_AndWhatIsSaid_WhatIsSaidWinning()
    {
        var carrying = new DistributionStep(["a", "b"], scaleBy: "chicks");

        var made = Catalog.Make("target.distribution", new JsonObject { ["columns"] = new JsonArray("w500", "w550") }, carrying);

        Assert.Equal(new DistributionStep(["w500", "w550"], scaleBy: "chicks"), made);
    }

    [Fact]
    public void WhatTheVerbDoesNotTake_IsNotCarried()
    {
        var made = Catalog.Make("target.labels", new JsonObject(), new DistributionStep(Bands, scaleBy: "chicks"));

        Assert.Equal(new LabelsStep(Bands), made);
    }

    [Fact]
    public void NothingSaidAndNothingCarried_IsTheVerbsOwnTemplate()
    {
        Assert.Equal(Catalog.ReadStep(Catalog.Describe("target").Template), Catalog.Make("target", new JsonObject(), carrying: null));
    }

    [Fact]
    public void AStepMadeUnderItsOwnVerb_FromItselfWithNothingSaid_IsTheStep()
    {
        var ahead = new AheadStep("Close", 5, AheadAs.Return);

        Assert.Equal(ahead, Catalog.Make("target.ahead", new JsonObject(), ahead));
    }

    [Fact]
    public void TheVerbsOwnRules_Hold()
    {
        var one = Assert.Throws<PipelineFileException>(
            () => Catalog.Make("target.distribution", new JsonObject { ["columns"] = new JsonArray("w500") }, carrying: null));

        Assert.Contains("at least two columns", one.Message, StringComparison.Ordinal);
        Assert.Throws<PipelineFileException>(() => Catalog.Make("target.nothing", new JsonObject(), carrying: null));
        Assert.Throws<PipelineFileException>(() => Catalog.Make("target", new JsonObject { ["colour"] = "red" }, carrying: null));
    }
}
