// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A pipeline also travels inside a larger file — a trained network and the pipeline it was trained behind are one file —
/// as the value of one of its keys. It is read there exactly as its own file is read, and every fault is placed where it
/// stands in the file a person opens: a fault on that file's line 313 is on line 313, not on line 3 of a pipeline cut
/// out of it.
/// </summary>
public class PipelineInPlaceTests
{
    private const string Faulty = """
        {"version": 3, "declaration": [
          {"step": "read.csv", "path": "titanic.csv"},
          {"step": "declare", "remainder": "drop", "columns": [{"name": "fare", "kind": "number", "optional": false}]},
          {"step": "normalise", "column": "fare", "scale": "sideways", "outOfRange": "pass"}
        ]}
        """;

    // A file around a pipeline: its own version, a network of so many lines before the pipeline, and the pipeline last,
    // its first line on the line of its key.
    private static string Around(string pipeline, int before) =>
        "{\n  \"version\": 1,\n  \"network\": {\n"
        + string.Join(",\n", Enumerable.Range(0, before).Select(at => $"    \"w{at}\": {at}.5"))
        + "\n  },\n  \"pipeline\": " + pipeline.ReplaceLineEndings("\n") + "\n}\n";

    private static PipelineFileException Refused(string json) =>
        Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn(), "pipeline"));

    [Fact]
    public void APipelineReadInPlace_IsThePipelineItsOwnFileHolds()
    {
        var trained = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("age")
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        var loaded = PreparedData.FromJson(Around(trained.ToJson(), 3), StepCatalog.BuiltIn(), "pipeline");

        Assert.Equal(trained.Declaration, loaded.Declaration);
        Assert.Equal(trained.Fitted.Keys, loaded.Fitted.Keys);
        Assert.Equal(trained.ToJson(), loaded.ToJson());
        Assert.Equal(0, loaded.Table.RowCount);
    }

    [Fact]
    public void AFaultOnTheFilesLine313_IsReportedOnLine313()
    {
        var fault = Assert.Single(Refused(Around(Faulty, 305)).Faults);

        Assert.Equal(313, fault.Line);
        Assert.Equal(3, fault.Column);
        Assert.Contains("Step 3", fault.Message, StringComparison.Ordinal);
        Assert.Contains("sideways", fault.Message, StringComparison.Ordinal);

        // The same pipeline in a file of its own says line 4.
        Assert.Equal(4, Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(Faulty, StepCatalog.BuiltIn())).Faults).Line);
    }

    [Fact]
    public void AKeyOfThePipeline_NotItsOwnOrWrittenTwice_IsPlacedInTheFileAroundIt()
    {
        const string colour = """
            {"version": 3,
             "colour": "red",
             "declaration": []}
            """;
        const string twice = """
            {"version": 3,
             "declaration": [], "declaration": []}
            """;

        var foreign = Assert.Single(Refused(Around(colour, 10)).Faults);

        Assert.Equal((16, 2), (foreign.Line, foreign.Column));
        Assert.Contains("'colour'", foreign.Message, StringComparison.Ordinal);

        var repeated = Assert.Single(Refused(Around(twice, 10)).Faults);

        Assert.Equal((16, 21), (repeated.Line, repeated.Column));
        Assert.Contains("written twice", repeated.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALargerFileThatIsNotOneJsonObjectHoldingThePipeline_IsRefusedWhereItIsWrong()
    {
        // Not JSON at all, after the pipeline: the whole file is read, since nothing in a file that is not JSON can be trusted.
        var broken = Refused("{\n  \"pipeline\": {\"declaration\": []},\n  \"network\": {\n");

        Assert.Contains("stops being JSON", Assert.Single(broken.Faults).Message, StringComparison.Ordinal);
        Assert.Equal(4, broken.Faults[0].Line);

        // A list, where a file holding a pipeline is an object.
        var list = Assert.Single(Refused("[{\"declaration\": []}]").Faults);

        Assert.Equal((1, 1), (list.Line, list.Column));
        Assert.Contains("'pipeline'", list.Message, StringComparison.Ordinal);

        // No pipeline there, or two.
        var none = Assert.Single(Refused("{\"version\": 1,\n \"network\": {}}").Faults);

        Assert.Equal((1, 1), (none.Line, none.Column));
        Assert.Contains("no 'pipeline'", none.Message, StringComparison.Ordinal);

        var twice = Assert.Single(Refused("{\"pipeline\": {\"declaration\": []},\n \"pipeline\": {\"declaration\": []}}").Faults);

        Assert.Equal((2, 2), (twice.Line, twice.Column));
        Assert.Contains("written twice", twice.Message, StringComparison.Ordinal);

        // A pipeline that is not an object is refused where it stands, as its own file would be.
        var three = Assert.Single(Refused("{\"network\": {},\n  \"pipeline\": 3}").Faults);

        Assert.Equal((2, 15), (three.Line, three.Column));
        Assert.Contains("one JSON object", three.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineOfANewerVersion_IsRefusedWhole_AtItsVersion()
    {
        var fault = Assert.Single(Refused(Around("""{"version": 9, "declaration": [{"step": "anything"}]}""", 2)).Faults);

        Assert.Equal((7, 16), (fault.Line, fault.Column));
        Assert.Contains("version 9", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheKeyAPipelineStandsUnderWrittenTwice_IsShownAsWordsAndNothingElse()
    {
        // The key is whoever wrote the file's, so a refusal that echoes it shows it the way every other refusal does: a
        // name holding a line break cannot start a line of its own in a log.
        const string key = "pipe\nline";

        var twice = Assert.Single(
            Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(
                "{\"pipe\\nline\": {\"declaration\": []},\n \"pipe\\nline\": {\"declaration\": []}}", StepCatalog.BuiltIn(), key)).Faults);

        Assert.Contains("written twice", twice.Message, StringComparison.Ordinal);
        Assert.Contains(@"'pipe\nline'", twice.Message, StringComparison.Ordinal);
        Assert.DoesNotContain(key, twice.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineBehindATokenWiderThanAPieceOfTheFile_IsReadAsItAlwaysWas()
    {
        // The file around a pipeline is read a piece of its UTF-8 at a time, and a piece is 65 536 bytes. One word longer
        // than that is a token the piece cannot hold, so the walk makes room for it rather than stopping short of it.
        var wide = "{\n  \"note\": \"" + new string('m', 70_000) + "\",\n  \"pipeline\": "
            + """{"version": 3, "declaration": [{"step": "read.csv", "path": "a.csv"}]}""" + "\n}\n";

        var read = PreparedData.FromJson(wide, StepCatalog.BuiltIn(), "pipeline");

        Assert.Equal("read.csv", Assert.Single(read.Declaration.Steps).Verb);
    }

    [Fact]
    public void AFaultBehindLettersOfMoreThanOneByte_IsAtItsColumnInCharacters()
    {
        // A place is counted in the bytes of UTF-8, and a column is shown in characters: a line of letters each written in
        // three or four bytes would otherwise put the fault several columns past where a person sees it.
        var fault = Assert.Single(Refused(
            "{\n  \"note\": \"漢字漢字\U0001F600\",\n  \"pipeline\": {\"version\": 3,\n   \"colour\": \"red\",\n   \"declaration\": []}\n}\n").Faults);

        Assert.Equal((4, 4), (fault.Line, fault.Column));
        Assert.Contains("'colour'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheKeyAPipelineStandsUnder_IsNamed()
    {
        Assert.Throws<ArgumentException>(() => PreparedData.FromJson("{}", StepCatalog.BuiltIn(), " "));
        Assert.Throws<ArgumentNullException>(() => PreparedData.FromJson("{}", StepCatalog.BuiltIn(), null!));
        Assert.Throws<ArgumentNullException>(() => PreparedData.FromJson(null!, StepCatalog.BuiltIn(), "pipeline"));
        Assert.Throws<ArgumentNullException>(() => PreparedData.FromJson("{}", null!, "pipeline"));

        var elsewhere = PreparedData.FromJson("""{"model": {"version": 3, "declaration": [{"step": "read.csv", "path": "a.csv"}]}}""", StepCatalog.BuiltIn(), "model");

        Assert.Equal("read.csv", Assert.Single(elsewhere.Declaration.Steps).Verb);
    }
}
