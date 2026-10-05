// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The two named courses, each filled in for the published data it was made for: the passenger list and the price series.
/// </summary>
/// <remarks>
/// Said once for every test that holds a course to the rules, since a course that is not filled in names no column and so
/// can be held to none of them.
/// </remarks>
internal static class FilledCourses
{
    private static JsonObject Json(string text) => JsonNode.Parse(text)!.AsObject();

    // The passengers: a table of people, with a gap in the ages and a fare the bounds of which are known.
    /// <summary>The table course, filled in for the passenger list.</summary>
    /// <returns>The course, with everything that names something of the passengers said.</returns>
    public static PipelineCourse Passengers() => PipelineCourse.Table
        .Say("read.csv", new JsonObject { ["path"] = Repository.Data("titanic.csv") })
        .Say("declare", Json("""
            {"columns":[{"name":"survived","kind":"integer","optional":false},{"name":"sibsp","kind":"integer","optional":false},
                        {"name":"parch","kind":"integer","optional":false},{"name":"fare","kind":"number","optional":false},
                        {"name":"age","kind":"number","optional":true}]}
            """))
        .Say("settle.gaps", Json("""{"column":"fare"}"""))
        .Say("feature.add", Json("""{"column":"family","left":"sibsp","arithmetic":"plus","right":"parch"}"""))
        .Say("scale.given", Json("""{"column":"fare","lowest":0,"highest":512}"""))
        .Say("split.stratified", Json("""{"column":"survived"}"""))
        .Say("target", Json("""{"column":"survived"}"""))
        .Say("drop.columns", Json("""{"columns":["sibsp","parch"]}"""))
        .Say("fill.missing", Json("""{"column":"age"}"""))
        .Say("normalise", Json("""{"column":"age"}"""));

    // The prices: a series in time, an answer read from the day after, and so a gap a day wide where the rows are divided.
    /// <summary>The series course, filled in for the price series.</summary>
    /// <returns>The course, with everything that names something of the prices said.</returns>
    public static PipelineCourse Prices() => PipelineCourse.SeriesInTime
        .Say("read.csv", new JsonObject { ["path"] = Repository.Data("apple.csv") })
        .Say("declare", Json("""
            {"columns":[{"name":"Date","kind":"timestamp","optional":false},{"name":"AAPL.High","kind":"number","optional":false},
                        {"name":"AAPL.Low","kind":"number","optional":false},{"name":"AAPL.Close","kind":"number","optional":false},
                        {"name":"AAPL.Volume","kind":"number","optional":false}]}
            """))
        .Say("order.by", Json("""{"columns":["Date"]}"""))
        .Say("settle.gaps", Json("""{"column":"AAPL.Volume"}"""))
        .Say("feature.add", Json("""{"column":"range","left":"AAPL.High","right":"AAPL.Low"}"""))
        .Say("scale.given", Json("""{"column":"AAPL.Volume","lowest":0,"highest":1000000000}"""))
        .Say("split.byTime", Json("""{"column":"Date"}"""))
        .Say("target.ahead", Json("""{"column":"AAPL.Close"}"""))
        .Say("drop.columns", Json("""{"columns":["AAPL.High","AAPL.Low"]}"""))
        .Say("fill.missing", Json("""{"column":"AAPL.Close"}"""))
        .Say("normalise", Json("""{"column":"AAPL.Close"}"""));
}
