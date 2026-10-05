// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// The wiki's page on starting from a course lists the steps of each course and what each waits for, and a list written
/// beside the code it describes is a list that drifts from it. So each table is held to the course it names: the same
/// verbs in the same order, and for each the keys its verb leaves to the person, in the order the verb writes them.
/// </summary>
public sealed partial class CourseWikiPageTests
{
    private static readonly StepCatalog Catalog = StepCatalog.BuiltIn().WithNetworks();

    [GeneratedRegex("^\\| (?<at>\\d+) \\| `(?<verb>[^`]+)` \\| (?<waits>[^|]*?) \\|", RegexOptions.Multiline)]
    private static partial Regex Row();

    /// <summary>One row of a course's table on the page.</summary>
    /// <param name="Verb">The verb the row names.</param>
    /// <param name="Waits">What the page says the step waits for, as it is written there.</param>
    private readonly record struct PageRow(string Verb, string Waits);

    private static PageRow[] TableUnder(string heading)
    {
        var page = File.ReadAllText(Path.Join(Wiki.Folder, "Starting-from-a-course.md")).ReplaceLineEndings("\n");
        var from = page.IndexOf($"\n## {heading}\n", StringComparison.Ordinal);

        Assert.True(from >= 0, $"The page has no section '{heading}'.");

        var section = page[from..];
        var next = section.IndexOf("\n## ", 1, StringComparison.Ordinal);

        return [.. Row().Matches(next < 0 ? section : section[..next]).Select(match => new PageRow(match.Groups["verb"].Value, match.Groups["waits"].Value.Trim()))];
    }

    private static string Waits(StepCatalog catalog, string verb)
    {
        var waiting = catalog.Describe(verb).Waiting;

        return waiting.Count == 0 ? "nothing" : string.Join(", ", waiting.Select(key => $"`{key}`"));
    }

    [Fact]
    public void TheTableOfTheTablesCourse_ListsItsStepsInOrder_AndWhatEachWaitsFor() =>
        Assert.Equal(
            [.. PipelineCourse.Table.Steps.Select(step => new PageRow(step.Verb, Waits(Catalog, step.Verb)))],
            TableUnder("The course for a table"));

    [Fact]
    public void TheTableOfTheSeriesCourse_ListsItsStepsInOrder_AndWhatEachWaitsFor() =>
        Assert.Equal(
            [.. PipelineCourse.SeriesInTime.Steps.Select(step => new PageRow(step.Verb, Waits(Catalog, step.Verb)))],
            TableUnder("The course for a series in time"));
}
