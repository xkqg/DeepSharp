// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One line says how the extremes of many columns are held, and the ways of finding the bounds are the methods.
/// </summary>
/// <remarks>
/// Four columns that all want their tails held were four lines that differed in a name and a distance. The line now names
/// how the bounds are worked out and lists the columns it holds for, and what reaches the declaration is what it always
/// was: one step a column. What happens to a value outside the bounds is a fact about one column — a refusal belongs on
/// the column that must never see one — so it stays on the verb that takes one.
/// </remarks>
public class ClipOutliersLineTests
{
    private static FittingBuilder Split() =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema
                .Integer("survived")
                .Number("fare", "volume", "trades")
                .Optional("age", ColumnKind.Number))
            .SplitStratified("survived", train: 0.70, validation: 0.15);

    private static ClipOutliersStep[] Clipped(FittingBuilder chain) =>
        [.. chain.Declaration.Steps.OfType<ClipOutliersStep>()];

    private static MethodInfo[] Offered() =>
        typeof(BoundsLine).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

    [Fact]
    public void OneBoundsLine_HoldsEachColumnTheWayItsMethodNames()
    {
        var clipped = Clipped(Split()
            .ClipOutliers(clip => clip
                .Iqr("fare", "age")
                .Sigma(2.5, "volume")
                .Quantile(0.01, "trades")));

        Assert.Equal(["fare", "age", "volume", "trades"], clipped.Select(step => step.Column));
        Assert.Equal([Bounds.Iqr, Bounds.Iqr, Bounds.Sigma, Bounds.Quantile], clipped.Select(step => step.Bounds));
        var written = new ClipOutliersStep("fare").At;

        Assert.Equal([written, written, 2.5, 0.01], clipped.Select(step => step.At));
        Assert.All(clipped, step => Assert.Equal(Outlier.Clip, step.Outlier));
    }

    [Fact]
    public void AKindThatNamesNoDistance_HoldsItWhereAColumnOnItsOwnIsHeld()
    {
        // Said once, by the step: the line leaves the distance out and so gets whatever the step says it is.
        var written = new ClipOutliersStep("fare").At;

        var clipped = Clipped(Split().ClipOutliers(clip => clip.Iqr("fare").Sigma("age")));

        Assert.Equal([written, written], clipped.Select(step => step.At));
    }

    [Fact]
    public void TheOneLineAndTheLinePerColumn_DeclareTheSameSteps()
    {
        // The door is the only thing that changes: what the declaration holds is step for step what it held.
        var one = Split()
            .ClipOutliers(clip => clip.Iqr("fare").Sigma(2.5, "volume").Quantile(0.01, "trades"))
            .Declaration;

        var each = Split()
            .ClipOutliers("fare")
            .ClipOutliers("volume", Bounds.Sigma, 2.5)
            .ClipOutliers("trades", Bounds.Quantile, 0.01)
            .Declaration;

        Assert.Equal(each.Steps, one.Steps);
        Assert.Equal(each.ToJson(), one.ToJson());
    }

    [Fact]
    public void EveryWayOfFindingTheBounds_HasAMethodOfItsOwnName_ThatTakesTheDistanceFirst()
    {
        // Open-closed: a kind of bounds added to the list that the line has no method for fails here, instead of being
        // reachable one column at a time only.
        Type[] distanceFirst = [typeof(double), typeof(string[])];

        foreach (var bounds in Enum.GetValues<Bounds>())
        {
            var method = Assert.Single(
                Offered(),
                candidate => candidate.Name == bounds.ToString()
                    && candidate.GetParameters().Select(parameter => parameter.ParameterType).SequenceEqual(distanceFirst));

            var written = Assert.Single(Clipped(Split().ClipOutliers(line => method.Invoke(line, [0.25, new[] { "fare" }]))));

            Assert.Equal(bounds, written.Bounds);
            Assert.Equal(0.25, written.At);
        }
    }

    [Fact]
    public void ASpreadAndTheMiddleHalf_CanLeaveTheDistanceUnsaid()
    {
        Assert.Contains(Offered(), method => method.Name == nameof(Bounds.Iqr) && method.GetParameters().Length == 1);
        Assert.Contains(Offered(), method => method.Name == nameof(Bounds.Sigma) && method.GetParameters().Length == 1);
    }

    [Fact]
    public void AQuantile_HasNoLineThatLeavesTheDistanceUnsaid()
    {
        // A distance for the spread is a multiple, and one and a half of them is a sensible way to start; for a quantile it
        // is a share of the column, and the verb for one column refuses it at that value. A line that offered the quantile
        // without a share could only ever fail, so it is not written.
        Assert.Throws<ArgumentOutOfRangeException>(() => Split().ClipOutliers("fare", Bounds.Quantile));

        var quantiles = Offered().Where(method => method.Name == nameof(Bounds.Quantile)).ToArray();

        Assert.NotEmpty(quantiles);
        Assert.All(
            quantiles,
            method => Assert.Contains(method.GetParameters(), parameter => parameter.ParameterType == typeof(double)));
    }

    [Fact]
    public void WhatHappensToAValueOutsideTheBounds_IsNotOnTheLine()
    {
        // That choice is a fact about one column, so it stays on the verb that takes one column rather than becoming a
        // decision the whole group would have to share.
        Assert.NotEmpty(Offered());

        Assert.DoesNotContain(
            Offered(),
            method => method.GetParameters().Any(parameter => parameter.ParameterType == typeof(Outlier)));
    }

    [Fact]
    public void ADistanceTheBoundsCannotUse_IsRefusedWhereTheLineIsWritten()
    {
        // The line has no rule of its own about it: the step refuses what it refuses, as it does for one column.
        Assert.Throws<ArgumentOutOfRangeException>(() => Split().ClipOutliers(clip => clip.Quantile(0.5, "fare")));
        Assert.Throws<ArgumentOutOfRangeException>(() => Split().ClipOutliers(clip => clip.Iqr(0, "fare")));
    }

    [Fact]
    public void AnEmptyBoundsLine_IsRefusedWhereItIsWritten()
    {
        // A line that holds nothing is a line somebody meant to finish.
        Assert.Throws<ArgumentException>(() => Split().ClipOutliers(clip => { }));
    }
}
