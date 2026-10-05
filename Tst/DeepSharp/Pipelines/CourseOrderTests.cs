// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The order a course teaches, held to the two things that decide it. The rules fix very little — a source first, then the
/// columns — and everything else is a flow somebody has to have taught: gaps before the features worked out from them, what
/// learns nothing before the split and what learns below it. So the order is written down once, by the course, and these
/// tests hold it both to the rules, by filling it in and asking, and to the flow, by saying each thing it teaches and why.
/// </summary>
public class CourseOrderTests
{
    private static readonly StepCatalog Catalog = Shipped.Catalog();

    // A before B, said by a verb or by a family of verbs ("split.*"), with the reason the course puts them so.
    private readonly record struct Taught(string Before, string After, string Because);

    private static readonly Taught[] TheFlow =
    [
        new("read.*", "declare", "everything works on columns, so the source comes first and the columns are declared at once after it"),
        new("declare", "settle.gaps", "a step reads the columns the schema declared, so none may stand above it"),
        new("settle.gaps", "feature.add", "a feature worked out from a column with a gap is itself a gap, so the gaps are settled first"),
        new("feature.add", "scale.given", "a feature is worked out from the columns as they stand, and scaled afterwards, to where a model takes it"),
        new("scale.given", "split.*", "scaling from bounds you know learns nothing from the rows, so it stands above the split"),
        new("split.*", "target*", "the answer stands right below the split, where every answer can stand: a return is made from its column as it was read"),
        new("target*", "drop.columns", "a column is dropped after the steps that read it, and the answer reads some"),
        new("drop.columns", "fill.missing", "what is not needed is dropped before the steps that learn from the rows are asked to work on it"),
        new("fill.missing", "normalise", "a gap is filled before the column is scaled, so the scale is learned from numbers"),
        new("normalise", "evidence.report", "the report measures the rows as they became"),
        new("evidence.report", "learn.network", "what a model is held to is said before the model is named"),
    ];

    private static int Place(PipelineCourse course, string verbOrFamily) =>
        verbOrFamily.EndsWith('*')
            ? FirstWhere(course, verb => verb.StartsWith(verbOrFamily[..^1], StringComparison.Ordinal))
            : FirstWhere(course, verb => verb == verbOrFamily);

    private static int FirstWhere(PipelineCourse course, Func<string, bool> matches)
    {
        for (var at = 0; at < course.Steps.Count; at++)
        {
            if (matches(course.Steps[at].Verb))
            {
                return at;
            }
        }

        return -1;
    }

    // Each adjacent pair of a filled course whose swap breaks a rule: the order the rules themselves fix.
    private static IReadOnlyList<Taught> ForcedByTheRules(PipelineCourse filled)
    {
        var steps = filled.ToDeclaration(Catalog).Steps;
        List<Taught> forced = [];

        for (var at = 0; at + 1 < steps.Count; at++)
        {
            IPipelineStep[] swapped = [.. steps];
            var first = swapped[at];

            swapped[at] = swapped[at + 1];
            swapped[at + 1] = first;

            if (PipelineDeclaration.FaultsIn(swapped).Count > 0)
            {
                forced.Add(new Taught(steps[at].Verb, steps[at + 1].Verb, string.Empty));
            }
        }

        return forced;
    }

    [Fact]
    public void EveryStepOfBothCourses_IsAVerbTheCatalogKnows_AndEachIsHeldOnce()
    {
        foreach (var course in new[] { PipelineCourse.Table, PipelineCourse.SeriesInTime })
        {
            Assert.All(course.Steps, step => Assert.True(Catalog.Knows(step.Verb), step.Verb));
            Assert.Equal(course.Steps.Count, course.Steps.Select(step => step.Verb).Distinct().Count());
        }
    }

    [Fact]
    public void TheCourseForATable_TeachesTheFlow_InThisOrder() =>
        Assert.Equal(
            ["read.csv", "declare", "settle.gaps", "feature.add", "scale.given", "split.stratified", "target", "drop.columns", "fill.missing", "normalise", "evidence.report", "learn.network"],
            PipelineCourse.Table.Steps.Select(step => step.Verb));

    [Fact]
    public void TheCourseForASeriesInTime_TeachesTheSameFlow_WithTheThreeDifferencesTheRulesMake() =>
        Assert.Equal(
            ["read.csv", "declare", "order.by", "settle.gaps", "feature.add", "scale.given", "split.byTime", "target.ahead", "drop.columns", "fill.missing", "normalise", "evidence.report", "learn.network"],
            PipelineCourse.SeriesInTime.Steps.Select(step => step.Verb));

    [Fact]
    public void EveryThingTheFlowTeaches_IsTheOrderOfBothCourses()
    {
        foreach (var course in new[] { PipelineCourse.Table, PipelineCourse.SeriesInTime })
        {
            Assert.All(TheFlow, taught =>
            {
                var before = Place(course, taught.Before);
                var after = Place(course, taught.After);

                Assert.True(before >= 0 && after >= 0, $"{taught.Before} and {taught.After} are both in the course: {taught.Because}");
                Assert.True(before < after, $"{taught.Before} comes before {taught.After}: {taught.Because}");
            });
        }
    }

    [Fact]
    public void ASeriesIsPutInOrder_RightAfterTheColumns_BeforeAnythingReadsTheOrder()
    {
        var series = PipelineCourse.SeriesInTime;

        Assert.Equal(Place(series, "declare") + 1, Place(series, "order.by"));
        Assert.Equal(Place(series, "order.by") + 1, Place(series, "settle.gaps"));
    }

    [Fact]
    public void TheRulesFixOnlyTheSourceAndTheColumns_AndTheRestOfTheOrderIsTaught()
    {
        // Measured by swapping each neighbouring pair of a filled course and asking the rules: only the source and the
        // schema cannot trade places, which is why the order is a thing a course has to say rather than one the rules give.
        var table = ForcedByTheRules(FilledCourses.Passengers());
        var series = ForcedByTheRules(FilledCourses.Prices());

        Assert.Equal(["read.csv > declare", "declare > settle.gaps"], table.Select(pair => $"{pair.Before} > {pair.After}"));
        Assert.Equal(["read.csv > declare", "declare > order.by", "split.byTime > target.ahead"], series.Select(pair => $"{pair.Before} > {pair.After}"));
    }

    [Fact]
    public void ACourseNobodyHasFilledIn_HasNoRuleFaultAtAnyStageOfFillingItIn()
    {
        // A block that waits for something does not read, and so takes no part in the pipeline the blocks above it make: the
        // rules are asked of the steps from the top that read, and of none below the first that does not. Filled in from the
        // top, one step at a time, there is never a fault on a step nobody has touched.
        foreach (var filled in new[] { FilledCourses.Passengers(), FilledCourses.Prices() })
        {
            for (var done = 0; done <= filled.Steps.Count; done++)
            {
                var readable = new List<IPipelineStep>();

                for (var at = 0; at < filled.Steps.Count; at++)
                {
                    var text = at < done ? filled.Steps[at].Skeleton(Catalog) : CourseStep.Of(filled.Steps[at].Verb).Skeleton(Catalog);

                    if (!TryRead(text, out var step))
                    {
                        break;
                    }

                    readable.Add(step);
                }

                Assert.Empty(PipelineDeclaration.FaultsIn(readable));
            }
        }
    }

    private static bool TryRead(string text, out IPipelineStep step)
    {
        try
        {
            step = Catalog.ReadStep(text);

            return true;
        }
        catch (PipelineFileException)
        {
            step = null!;

            return false;
        }
    }
}
