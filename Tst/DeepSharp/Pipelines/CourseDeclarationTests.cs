// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A course filled in is a pipeline, and a course not filled in is refused whole, naming every step that still waits and
/// what it waits for. Nothing a person has not said is ever made up for them: a step that names something of theirs is not
/// started from an example, because an example reads as well as a decision.
/// </summary>
public class CourseDeclarationTests
{
    private static readonly StepCatalog Catalog = Shipped.Catalog();

    private static JsonObject Json(string text) => JsonNode.Parse(text)!.AsObject();

    [Fact]
    public void ACourseNobodyFilledIn_WaitsAtEveryStepThatNamesSomething_AndAtNoOther()
    {
        var waiting = PipelineCourse.Table.Waiting(Catalog);

        Assert.Equal(
            ["read.csv", "declare", "settle.gaps", "feature.add", "scale.given", "split.stratified", "target", "drop.columns", "fill.missing", "normalise"],
            waiting.Select(fault => fault.Verb));
        Assert.Equal([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], waiting.Select(fault => fault.At));
        Assert.Contains("'path'", waiting[0].Message, StringComparison.Ordinal);
        Assert.Contains("'column', 'lowest', 'highest'", waiting[4].Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepThatWaitsForNothing_IsNeverListedAsWaiting() =>
        Assert.DoesNotContain(PipelineCourse.Table.Waiting(Catalog), fault => fault.Verb is "evidence.report" or "learn.network");

    [Fact]
    public void ACourseFilledInAllTheWay_WaitsForNothing()
    {
        Assert.Empty(FilledCourses.Passengers().Waiting(Catalog));
        Assert.Empty(FilledCourses.Prices().Waiting(Catalog));
    }

    [Fact]
    public void ACourseNotFilledIn_IsRefusedWhole_NamingEveryStepThatWaits()
    {
        var refused = Assert.Throws<DeclarationException>(() => PipelineCourse.Table.ToDeclaration(Catalog));

        Assert.Equal(10, refused.Faults.Count);
        Assert.Contains("Step 1, 'read.csv'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("Step 10, 'normalise'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACourseFilledInHalfWay_IsRefusedAtTheStepsStillWaiting_AndNotAtTheOnesFilledIn()
    {
        var half = PipelineCourse.Table.Say("read.csv", new JsonObject { ["path"] = Repository.Data("titanic.csv") });

        var refused = Assert.Throws<DeclarationException>(() => half.ToDeclaration(Catalog));

        Assert.Equal(9, refused.Faults.Count);
        Assert.DoesNotContain(refused.Faults, fault => fault.Verb == "read.csv");
    }

    [Fact]
    public void ACourseNotFilledInAllTheWay_SaysEveryFaultTogether_WhatWaitsAndWhatIsRefused()
    {
        var course = PipelineCourse.Table.Say("scale.given", Json("""{"column":"fare","lowest":5,"highest":1}"""));

        var refused = Assert.Throws<DeclarationException>(() => course.ToDeclaration(Catalog));

        // Nine steps still wait, and the one that has everything said is refused by its verb: each at its own step, in order.
        Assert.Equal(10, refused.Faults.Count);
        Assert.Equal(refused.Faults.Select(fault => fault.At).Order(), refused.Faults.Select(fault => fault.At));
        Assert.Equal(9, refused.Faults.Count(fault => fault.Message.Contains("waits for", StringComparison.Ordinal)));
        Assert.Contains(
            refused.Faults,
            fault => fault.Verb == "scale.given" && fault.Message.Contains("lower value to a higher one", StringComparison.Ordinal));
    }

    [Fact]
    public void AKeyTheVerbDoesNotTake_IsRefusedAtItsStep_WhileTheStepStillWaits()
    {
        var course = PipelineCourse.Table.Say("read.csv", Json("""{"pth":"a.csv"}"""));

        var refused = Assert.Throws<DeclarationException>(() => course.ToDeclaration(Catalog));

        Assert.Contains(refused.Faults, fault => fault.At == 0 && fault.Message.Contains("'pth'", StringComparison.Ordinal));
        Assert.Contains(refused.Faults, fault => fault.At == 0 && fault.Message.Contains("waits for", StringComparison.Ordinal));
    }

    [Fact]
    public void ANameAStepMayLeaveOut_IsNeverMadeUpForIt()
    {
        // 'into' is the name of the column a step makes, and leaving it out means the column is worked on where it stands: the
        // template's example, a column called 'column', is not what the person said.
        var course = PipelineCourse.Of(
        [
            CourseStep.Of("read.csv", new JsonObject { ["path"] = Repository.Data("titanic.csv") }),
            CourseStep.Of("declare", Json("""{"columns":[{"name":"fare","kind":"number","optional":false}]}""")),
            CourseStep.Of("maths", Json("""{"column":"fare"}""")),
        ]);

        var declaration = course.ToDeclaration(Catalog);
        var written = System.Text.Encoding.UTF8.GetString(declaration.Steps[2].Canonical());

        Assert.DoesNotContain("\"into\":\"column\"", written, StringComparison.Ordinal);
        Assert.Contains("\"into\":\"fare\"", written, StringComparison.Ordinal);
    }

    [Fact]
    public void AVerbNothingHereKnows_IsRefusedAtItsStep()
    {
        var course = PipelineCourse.Of([CourseStep.Of("read.csv", Json("""{"path":"a.csv"}""")), CourseStep.Of("not.a.verb")]);

        var refused = Assert.Throws<DeclarationException>(() => course.ToDeclaration(Catalog));

        var fault = Assert.Single(refused.Faults);
        Assert.Equal(1, fault.At);
        Assert.Contains("not a step anything here knows", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AVerbAnotherPackageBrings_IsRefusedByACatalogNeverTaughtIt_NamingThePackage()
    {
        var refused = Assert.Throws<DeclarationException>(() => FilledCourses.Passengers().ToDeclaration(StepCatalog.BuiltIn()));

        var fault = Assert.Single(refused.Faults);
        Assert.Equal("learn.network", fault.Verb);
        Assert.Contains("DeepSharp.Learners.Networks", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AValueSaidThatTheVerbRefuses_IsRefusedAtItsStep_WithTheVerbsOwnWords()
    {
        var wrong = FilledCourses.Passengers().Say("scale.given", Json("""{"lowest":5,"highest":1}"""));

        var refused = Assert.Throws<DeclarationException>(() => wrong.ToDeclaration(Catalog));

        var fault = Assert.Single(refused.Faults);
        Assert.Equal(4, fault.At);
        Assert.Equal("scale.given", fault.Verb);
    }

    [Fact]
    public void ACourseOfTheStepsThatBreakARule_IsRefusedWithTheRule()
    {
        var split = FilledCourses.Passengers().Say("split.stratified", Json("""{"column":"sex"}"""));

        var refused = Assert.Throws<DeclarationException>(() => split.ToDeclaration(Catalog));

        Assert.Contains(refused.Faults, fault => fault.Message.Contains("'sex'", StringComparison.Ordinal));
    }

    [Fact]
    public void TheTableCourse_FilledInForThePassengers_IsThePipelineItNames_AndItRuns()
    {
        var declaration = FilledCourses.Passengers().ToDeclaration(Catalog);

        Assert.Equal(PipelineCourse.Table.Steps.Select(step => step.Verb), declaration.Steps.Select(step => step.Verb));

        var prepared = new Pipeline(declaration).Run();

        Assert.Equal(891, Enum.GetValues<Part>().Sum(prepared.CountIn));
        Assert.True(prepared.CountIn(Part.Train) > prepared.CountIn(Part.Validation));
    }

    [Fact]
    public void TheSeriesCourse_FilledInForThePrices_IsThePipelineItNames_AndItRuns()
    {
        var declaration = FilledCourses.Prices().ToDeclaration(Catalog);

        Assert.Equal(PipelineCourse.SeriesInTime.Steps.Select(step => step.Verb), declaration.Steps.Select(step => step.Verb));

        var prepared = new Pipeline(declaration).Run();

        Assert.True(prepared.CountIn(Part.Train) > 0);
        Assert.True(prepared.CountIn(Part.Test) > 0);
    }

    [Fact]
    public void ThePipelineAStartedCourseIs_IsTheDeclarationItMakes()
    {
        var pipeline = Pdd.From(FilledCourses.Passengers(), Catalog);

        Assert.Equal(FilledCourses.Passengers().ToDeclaration(Catalog), pipeline.Declaration);
    }

    [Fact]
    public void ACourseNotFilledIn_StartsNoPipeline() =>
        Assert.Throws<DeclarationException>(() => Pdd.From(PipelineCourse.SeriesInTime, Catalog));

    [Fact]
    public void WhatIsSaidOfAVerb_ReplacesWhatWasSaidOfThatKey_AndKeepsTheRest()
    {
        var course = PipelineCourse.SeriesInTime
            .Say("split.byTime", Json("""{"column":"Date","gap":3}"""))
            .Say("split.byTime", Json("""{"gap":5}"""));

        var split = course.Steps.Single(step => step.Verb == "split.byTime");

        Assert.Equal("""{"column":"Date","gap":5}""", split.Said);
    }

    [Fact]
    public void AKeySaidAsNothing_TakesBackWhatWasSaid_AndLeavesItWaiting()
    {
        var said = PipelineCourse.Table.Say("target", Json("""{"column":"survived"}"""));
        var unsaid = said.Say("target", Json("""{"column":null}"""));

        Assert.Empty(said.Steps.Single(step => step.Verb == "target").Waiting(Catalog));
        Assert.Equal(["column"], unsaid.Steps.Single(step => step.Verb == "target").Waiting(Catalog));
    }

    [Fact]
    public void SayingOfACourse_LeavesTheCourseItWasSaidOfAsItWas()
    {
        var before = PipelineCourse.Table;

        _ = before.Say("read.csv", new JsonObject { ["path"] = "a.csv" });

        Assert.Equal(PipelineCourse.Table, before);
        Assert.Equal("{}", before.Steps[0].Said);
    }

    [Fact]
    public void AVerbTheCourseDoesNotHold_CannotBeSaidOf_AndTheRefusalNamesTheVerbsItHolds()
    {
        var refused = Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Say("order.by", Json("""{"columns":["a"]}""")));

        Assert.Contains("order.by", refused.Message, StringComparison.Ordinal);
        Assert.Contains("read.csv", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AVerbTheCourseHoldsTwice_CannotBeSaidOfWithoutSayingWhich()
    {
        var twice = PipelineCourse.Of([CourseStep.Of("feature.add"), CourseStep.Of("feature.add")]);

        var refused = Assert.Throws<ArgumentException>(() => twice.Say("feature.add", Json("""{"column":"a"}""")));

        Assert.Contains("more than once", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SayingNothingOfAVerb_Or_OfNoVerb_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => PipelineCourse.Table.Say("target", null!));
        Assert.Throws<ArgumentException>(() => PipelineCourse.Table.Say(" ", new JsonObject()));
    }
}
