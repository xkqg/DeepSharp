// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One step of a course: a verb and what has been said of it so far. A step cannot be half made, so what a person has not
/// decided yet is not a step but the lack of one keyed by its verb — and what has been decided has to be kept as the
/// value it is, however the person who said it happened to order the keys.
/// </summary>
public class CourseStepTests
{
    [Fact]
    public void TwoStepsThatSayTheSameThing_AreTheSameStep_WhateverOrderTheKeysWereSaidIn()
    {
        var one = CourseStep.Of("split.stratified", new JsonObject { ["column"] = "survived", ["seed"] = 7 });
        var other = CourseStep.Of("split.stratified", new JsonObject { ["seed"] = 7, ["column"] = "survived" });

        Assert.Equal(one, other);
        Assert.Equal(one.GetHashCode(), other.GetHashCode());
        Assert.Equal(one.Said, other.Said);
    }

    [Fact]
    public void AStepKeepsWhatWasSaidWhenItWasMade_NotWhatTheCallerDoesToItAfterwards()
    {
        var said = new JsonObject { ["column"] = "age" };
        var step = CourseStep.Of("normalise", said);

        said["column"] = "fare";
        said["scale"] = "robust";

        Assert.Equal("""{"column":"age"}""", step.Said);
        Assert.Equal(CourseStep.Of("normalise", new JsonObject { ["column"] = "age" }), step);
    }

    [Fact]
    public void AKeySaidAsNothing_IsNotSaid()
    {
        // A file written by hand shows what still waits as a null, so a null is the same as leaving the key out.
        var written = CourseStep.Of("read.csv", new JsonObject { ["path"] = null });

        Assert.Equal(CourseStep.Of("read.csv"), written);
        Assert.Equal("{}", written.Said);
    }

    [Fact]
    public void ADifferentValueOrAnotherVerb_IsAnotherStep()
    {
        var age = CourseStep.Of("normalise", new JsonObject { ["column"] = "age" });

        Assert.NotEqual(age, CourseStep.Of("normalise", new JsonObject { ["column"] = "fare" }));
        Assert.NotEqual(age, CourseStep.Of("fill.missing", new JsonObject { ["column"] = "age" }));
        Assert.NotEqual(age, CourseStep.Of("normalise"));
    }

    [Fact]
    public void ANestedValue_IsKeptAsTheValueItIs()
    {
        var columns = new JsonObject
        {
            ["columns"] = new JsonArray(new JsonObject { ["name"] = "survived", ["kind"] = "integer" }),
        };

        var step = CourseStep.Of("declare", columns);

        Assert.Equal("""{"columns":[{"name":"survived","kind":"integer"}]}""", step.Said);
    }

    [Theory]
    [InlineData("")]
    [InlineData("  ")]
    public void AStepWithoutAVerb_IsRefused(string verb)
    {
        var refused = Assert.Throws<ArgumentException>(() => CourseStep.Of(verb));

        Assert.Contains("verb", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheVerbIsWhatAStepIsWrittenUnder_SoItIsNoKeyToSayOfIt()
    {
        var refused = Assert.Throws<ArgumentException>(() => CourseStep.Of("read.csv", new JsonObject { ["step"] = "read.json", ["path"] = "a.json" }));

        Assert.Contains("'step'", refused.Message, StringComparison.Ordinal);
        Assert.Equal("said", refused.ParamName);
    }

    [Fact]
    public void AStepWithoutAVerbAtAll_IsRefused() =>
        Assert.Throws<ArgumentNullException>(() => CourseStep.Of(null!));

    [Fact]
    public void ANullSaid_MeansNothingWasSaid() =>
        Assert.Equal(CourseStep.Of("read.csv"), CourseStep.Of("read.csv", said: null));
}
