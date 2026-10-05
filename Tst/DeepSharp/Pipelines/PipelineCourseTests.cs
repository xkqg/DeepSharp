// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A course: the steps of a prepared pipeline in the order they belong, each as much said as a person has said of it. It is
/// one value, equal to another that says the same, and it keeps what it was made of however its maker treats the list
/// afterwards.
/// </summary>
public class PipelineCourseTests
{
    private static readonly CourseStep Read = CourseStep.Of("read.csv", new JsonObject { ["path"] = "titanic.csv" });
    private static readonly CourseStep Declare = CourseStep.Of("declare");

    [Fact]
    public void ACourse_HoldsItsStepsInTheOrderTheyWereGiven()
    {
        var course = PipelineCourse.Of([Read, Declare]);

        Assert.Equal([Read, Declare], course.Steps);
    }

    [Fact]
    public void ACourse_KeepsTheStepsItWasMadeOf_NotTheListItsMakerKeepsChanging()
    {
        List<CourseStep> given = [Read, Declare];
        var course = PipelineCourse.Of(given);

        given.Clear();

        Assert.Equal(2, course.Steps.Count);
    }

    [Fact]
    public void TwoCourses_AreTheSameCourse_WhenTheySayTheSameStepsInTheSameOrder()
    {
        var one = PipelineCourse.Of([Read, Declare]);
        var same = PipelineCourse.Of([CourseStep.Of("read.csv", new JsonObject { ["path"] = "titanic.csv" }), CourseStep.Of("declare")]);

        Assert.Equal(one, same);
        Assert.Equal(one.GetHashCode(), same.GetHashCode());
        Assert.True(one.Equals((object)same));
    }

    [Fact]
    public void AnotherOrderAnotherStepOrAnotherLength_IsAnotherCourse()
    {
        var one = PipelineCourse.Of([Read, Declare]);

        Assert.NotEqual(one, PipelineCourse.Of([Declare, Read]));
        Assert.NotEqual(one, PipelineCourse.Of([Read, CourseStep.Of("settle.gaps")]));
        Assert.NotEqual(one, PipelineCourse.Of([Read]));
        Assert.False(one.Equals(null));
        Assert.False(one.Equals("a course"));
    }

    [Fact]
    public void ACourseWithNoStep_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => PipelineCourse.Of([]));
        Assert.Throws<ArgumentNullException>(() => PipelineCourse.Of(null!));
    }

    [Fact]
    public void AStepThatIsNothing_IsRefusedWhereTheCourseIsMade() =>
        Assert.Throws<ArgumentException>(() => PipelineCourse.Of([Read, null!]));

    [Fact]
    public void ACourseCannotBeToldTheVerbAsAKey_SoTheStepOfAnotherVerbCannotBeSmuggledIn() =>
        Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Say("read.csv", new JsonObject { ["step"] = "read.json" }));

    [Fact]
    public void AlsoAddsOneMoreStepOfAVerb_RightAfterTheLastOneOfIt()
    {
        var course = PipelineCourse.Table
            .Say("normalise", new JsonObject { ["column"] = "age" })
            .Also("normalise", new JsonObject { ["column"] = "fare" })
            .Also("normalise", new JsonObject { ["column"] = "family" });

        var at = course.Steps.Select(step => step.Verb).ToList().IndexOf("normalise");

        Assert.Equal(PipelineCourse.Table.Steps.Count + 2, course.Steps.Count);
        Assert.Equal(["normalise", "normalise", "normalise", "evidence.report"], course.Steps.Skip(at).Take(4).Select(step => step.Verb));
        Assert.Equal(["""{"column":"age"}""", """{"column":"fare"}""", """{"column":"family"}"""], course.Steps.Skip(at).Take(3).Select(step => step.Said));
    }

    [Fact]
    public void AlsoOfAVerbTheCourseDoesNotHold_IsRefused_NamingTheVerbsItHolds()
    {
        var refused = Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Also("order.by", new JsonObject()));

        Assert.Contains("order.by", refused.Message, StringComparison.Ordinal);
        Assert.Contains("read.csv", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AlsoNeedsAVerbAndWhatIsSaidOfIt()
    {
        Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Also(" ", new JsonObject()));
        Assert.Throws<ArgumentNullException>(() => PipelineCourse.Table.Also("normalise", null!));
    }

    [Fact]
    public void AddingAStep_LeavesTheCourseItWasAddedToAsItWas()
    {
        var before = PipelineCourse.Table;

        _ = before.Also("normalise", new JsonObject { ["column"] = "fare" });

        Assert.Equal(12, before.Steps.Count);
    }

    [Fact]
    public void WithoutLeavesOutEveryStepOfTheseVerbs()
    {
        var plain = PipelineCourse.Table.Without("settle.gaps", "feature.add", "scale.given");

        Assert.Equal(PipelineCourse.Table.Steps.Count - 3, plain.Steps.Count);
        Assert.Equal(PipelineCourse.Table.Steps.Where(step => step.Verb is not ("settle.gaps" or "feature.add" or "scale.given")), plain.Steps);
    }

    [Fact]
    public void WithoutNoVerbAtAll_IsTheCourseAgain() =>
        Assert.Equal(PipelineCourse.Table, PipelineCourse.Table.Without());

    [Fact]
    public void WithoutAVerbTheCourseDoesNotHold_IsRefused_NamingTheVerbsItHolds()
    {
        var refused = Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Without("order.by"));

        Assert.Contains("order.by", refused.Message, StringComparison.Ordinal);
        Assert.Contains("read.csv", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutEveryStep_IsRefused_BecauseACourseHoldsAtLeastOne()
    {
        var verbs = PipelineCourse.Table.Steps.Select(step => step.Verb).Distinct().ToArray();

        var refused = Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Without(verbs));

        Assert.Contains("no step", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutNeedsVerbsThatAreNamed()
    {
        Assert.Throws<ArgumentNullException>(() => PipelineCourse.Table.Without(null!));
        Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Without(" "));
    }

    [Fact]
    public void TheNamedCourses_AreTheTableAndTheSeriesInTime() =>
        Assert.Equal([PipelineCourse.Table, PipelineCourse.SeriesInTime], PipelineCourse.Named);

    [Fact]
    public void TheStepsOfACourse_CannotBeChangedThroughTheList() =>
        Assert.Throws<NotSupportedException>(() => ((IList<CourseStep>)PipelineCourse.Of([Read]).Steps).Add(Declare));
}
