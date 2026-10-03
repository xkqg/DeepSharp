// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One line says which cycle many moments are placed on, and the cycles are the methods.
/// </summary>
/// <remarks>
/// A start, an end and a creation time that each want an hour or a month on a circle were lines that differed in a column
/// name, and the only way to say a cycle was to repeat the whole verb. The line now names the cycle once and lists the
/// columns it holds for, and what reaches the declaration is what it always was: one step a column and cycle. How the two
/// values of a place are written down belongs to how a model reads them, so it stays on the verb that takes one column.
/// </remarks>
public class CyclicalLineTests
{
    private static PipelineBuilder Moments() =>
        Pdd.Create()
            .ReadCsv("rides.csv")
            .Declare(schema => schema.Timestamp("start", "end", "created"));

    private static CyclicalStep[] Placed(PipelineBuilder chain) =>
        [.. chain.Declaration.Steps.OfType<CyclicalStep>()];

    private static MethodInfo[] Offered() =>
        typeof(PeriodLine).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

    [Fact]
    public void OneCycleLine_PlacesEachColumnOnTheCycleTheMethodNames()
    {
        var placed = Placed(Moments()
            .Cyclical(period => period
                .HourOfDay("start", "end")
                .DayOfWeek("start")
                .MonthOfYear("created")));

        Assert.Equal(["start", "end", "start", "created"], placed.Select(step => step.Column));
        Assert.Equal(
            [Period.HourOfDay, Period.HourOfDay, Period.DayOfWeek, Period.MonthOfYear],
            placed.Select(step => step.Period));
    }

    [Fact]
    public void ACycleThatNamesNoForm_WritesThePlaceAsTheVerbForOneColumnDoes()
    {
        // Said once, by the step: the line leaves the form out and so gets whatever the step says it is.
        var written = new CyclicalStep("start", Period.HourOfDay).Form;

        var placed = Placed(Moments().Cyclical(period => period.HourOfDay("start").MonthOfYear("created")));

        Assert.Equal([written, written], placed.Select(step => step.Form));
    }

    [Fact]
    public void TheOneLineAndTheLinePerColumn_DeclareTheSameSteps()
    {
        // The door is the only thing that changes: what the declaration holds is step for step what it held.
        var one = Moments()
            .Cyclical(period => period.HourOfDay("start", "end").DayOfWeek("start").MonthOfYear("created"))
            .Declaration;

        var each = Moments()
            .Cyclical("start", Period.HourOfDay)
            .Cyclical("end", Period.HourOfDay)
            .Cyclical("start", Period.DayOfWeek)
            .Cyclical("created", Period.MonthOfYear)
            .Declaration;

        Assert.Equal(each.Steps, one.Steps);
        Assert.Equal(each.ToJson(), one.ToJson());
    }

    [Fact]
    public void EveryCycle_HasAMethodOfItsOwnName_ThatPlacesOnThatCycle()
    {
        // Open-closed: a cycle added to the list that the line has no method for fails here, instead of being reachable
        // one column at a time only.
        foreach (var period in Enum.GetValues<Period>())
        {
            var method = Assert.Single(Offered(), candidate => candidate.Name == period.ToString());

            var written = Assert.Single(Placed(Moments().Cyclical(line => method.Invoke(line, [new[] { "start" }]))));

            Assert.Equal(period, written.Period);
        }
    }

    [Fact]
    public void HowThePlaceIsWrittenDown_IsNotOnTheLine()
    {
        // The form belongs to how a model is to read the two values, so it stays on the verb that takes one column
        // rather than becoming a choice the whole group would have to share.
        Assert.NotEmpty(Offered());

        Assert.DoesNotContain(
            Offered(),
            method => method.GetParameters().Any(parameter => parameter.ParameterType == typeof(Form)));
    }

    [Fact]
    public void AnEmptyCycleLine_IsRefusedWhereItIsWritten()
    {
        // A line that places nothing is a line somebody meant to finish.
        Assert.Throws<ArgumentException>(() => Moments().Cyclical(period => { }));
    }
}
