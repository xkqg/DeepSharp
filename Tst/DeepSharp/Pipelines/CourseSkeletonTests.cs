// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The step a course starts with: the verb's own skeleton, every key it needs said present. What names something of yours — a
/// column, a file, the columns a schema declares, a bound — waits for you as nothing, because no example of it could be
/// right and one that merely reads would pass for a decision; what only settles how, starts as the verb starts it.
/// </summary>
public class CourseSkeletonTests
{
    private static readonly StepCatalog Catalog = Shipped.Catalog();

    private static IEnumerable<string> KeysOf(string step)
    {
        using var document = JsonDocument.Parse(step);

        return [.. document.RootElement.EnumerateObject().Select(property => property.Name).Where(name => name != StepCatalog.StepKey)];
    }

    [Fact]
    public void ASkeleton_HoldsTheKeysOfTheTemplate_ButForTheNamesAStepMayLeaveOut() =>
        // A key whose absence means something is left out of the template, and so of the skeleton. So is a name the step may
        // do without: an example of it would be taken for a decision, and leaving it out is how the step decides it itself.
        Assert.All(Catalog.Descriptions, description =>
        {
            var leftOut = description.Parameters
                .Where(parameter => parameter.IsLeftToThePerson && parameter.WaitingKeys.Count == 0)
                .SelectMany(parameter => parameter.Keys);

            Assert.Equal(
                KeysOf(description.Template).Except(leftOut).Order(StringComparer.Ordinal),
                KeysOf(description.Skeleton).Order(StringComparer.Ordinal));
        });

    [Fact]
    public void ANameAStepMayLeaveOut_IsLeftOutOfItsSkeleton_AndASettingThatOnlySettlesHowStartsAsItsVerbStarts()
    {
        Assert.Equal("""{"step":"maths","column":null,"maths":"log1p"}""", Catalog.Describe("maths").Skeleton);
        Assert.DoesNotContain("columns", KeysOf(Catalog.Describe("evidence.profile").Skeleton));

        // How far out the bounds sit is a setting, not a bound of the person's column.
        Assert.Equal("""{"step":"outliers.clip","column":null,"bounds":"iqr","at":1.5,"outlier":"clip"}""", Catalog.Describe("outliers.clip").Skeleton);
        Assert.Equal(["column"], Catalog.Describe("outliers.clip").Waiting);
    }

    [Theory]
    [InlineData("read.csv", """{"step":"read.csv","path":null}""")]
    [InlineData("declare", """{"step":"declare","remainder":"drop","columns":null}""")]
    [InlineData("feature.add", """{"step":"feature.add","column":null,"left":null,"arithmetic":"minus","right":null}""")]
    [InlineData("scale.given", """{"step":"scale.given","column":null,"lowest":null,"highest":null,"lands":"signed"}""")]
    [InlineData("split.stratified", """{"step":"split.stratified","column":null,"train":0.7,"validation":0.15,"test":0.15,"predict":0,"seed":20260923}""")]
    [InlineData("normalise", """{"step":"normalise","column":null,"scale":"midrange","outOfRange":"pass"}""")]
    [InlineData("target.ahead", """{"step":"target.ahead","column":null,"ahead":1,"as":"value"}""")]
    [InlineData("order.by", """{"step":"order.by","columns":null}""")]
    public void WhatNamesSomethingOfYours_WaitsAsNothing_AndTheRestStartsAsTheVerbStarts(string verb, string skeleton) =>
        Assert.Equal(skeleton, Catalog.Describe(verb).Skeleton);

    [Fact]
    public void AStepWithAKeyWaiting_IsRefusedByTheReader_AndTheRefusalNamesAKeyThatWaits() =>
        Assert.All(Catalog.Descriptions.Where(description => description.Waiting.Count > 0), description =>
        {
            var refused = Assert.Throws<PipelineFileException>(() => Catalog.ReadStep(description.Skeleton));

            Assert.Contains(
                refused.Faults,
                fault => description.Waiting.Any(key => fault.Message.Contains($"'{key}'", StringComparison.Ordinal)));
        });

    [Fact]
    public void AVerbWithNothingLeftToYou_StartsAsItsTemplate_AndReadsAsItStands() =>
        Assert.All(Catalog.Descriptions.Where(description => description.Parameters.All(parameter => !parameter.IsLeftToThePerson)), description =>
        {
            Assert.Equal(description.Template, description.Skeleton);
            Assert.Equal(description.Verb, Catalog.ReadStep(description.Skeleton).Verb);
        });

    [Fact]
    public void TheVerbsOfTheCourses_AreEachOneThatNamesSomethingOrOneThatNeedsNothing()
    {
        // The report and the network settle how, and name nothing of the rows: they start as the verb starts, and read.
        Assert.Empty(Catalog.Describe("evidence.report").Waiting);
        Assert.Empty(Catalog.Describe("learn.network").Waiting);
        Assert.Equal(["path"], Catalog.Describe("read.csv").Waiting);
        Assert.Equal(["columns"], Catalog.Describe("declare").Waiting);
        Assert.Equal(["column", "lowest", "highest"], Catalog.Describe("scale.given").Waiting);
    }

    [Fact]
    public void WhatHasBeenSaid_ReplacesWhatWaits_AndStaysWhereTheVerbPutsIt()
    {
        var step = CourseStep.Of("scale.given", new JsonObject { ["column"] = "fare", ["highest"] = 512 });

        Assert.Equal("""{"step":"scale.given","column":"fare","lowest":null,"highest":512,"lands":"signed"}""", step.Skeleton(Catalog));
        Assert.Equal(["lowest"], step.Waiting(Catalog));
    }

    [Fact]
    public void AStepWithEverythingSaid_WaitsForNothing_AndItsSkeletonIsTheStepItMakes()
    {
        var said = new JsonObject { ["column"] = "age", ["scale"] = "robust" };
        var step = CourseStep.Of("normalise", said);

        Assert.Empty(step.Waiting(Catalog));
        Assert.Equal(Catalog.Make("normalise", said, carrying: null), Catalog.ReadStep(step.Skeleton(Catalog)));
    }

    [Fact]
    public void AVerbNothingHereKnows_CannotSayWhatItWaitsFor()
    {
        var unknown = CourseStep.Of("not.a.verb");

        Assert.Throws<NotSupportedException>(() => unknown.Waiting(Catalog));
        Assert.Throws<NotSupportedException>(() => unknown.Skeleton(Catalog));
    }

    [Fact]
    public void WhatNamesSomethingOfTheirs_IsLeftToThePerson_AndWhatSettlesHow_IsNot()
    {
        Assert.True(new ColumnParameter("column", "What it reads.", "column", ColumnKinds.Any).IsLeftToThePerson);
        Assert.True(new ColumnsParameter("columns", "Which columns.", ["a"], ColumnKinds.Any).IsLeftToThePerson);
        Assert.True(new NewColumnParameter("into", "Where it goes.", "column").IsLeftToThePerson);
        Assert.True(new FilePathParameter("path", "Where the rows are.", "data.csv").IsLeftToThePerson);
        Assert.True(new NumberParameter("lowest", "The lowest.", 0) { IsABound = true }.IsLeftToThePerson);
        Assert.False(new NumberParameter("at", "How far out.", 1.5).IsLeftToThePerson);
        Assert.False(new WholeNumberParameter("seed", "Where the draws start.", 1).IsLeftToThePerson);
        Assert.False(new TrueOrFalseParameter("flag", "Whether.", example: true).IsLeftToThePerson);
        Assert.False(new ShareParameter("share", "How much.").IsLeftToThePerson);
        Assert.False(new TextParameter("sheet", "Which sheet.", "Sheet1").IsLeftToThePerson);
    }

    [Fact]
    public void AnOptionalColumn_WaitsForNothing_BecauseLeavingItOutMeansSomething()
    {
        var optional = new ColumnParameter("by", "What it divides by.", ColumnKinds.Any);

        Assert.True(optional.IsLeftToThePerson);
        Assert.Empty(optional.WaitingKeys);
        Assert.Equal(["column"], new ColumnParameter("column", "What it reads.", "column", ColumnKinds.Any).WaitingKeys);
    }
}
