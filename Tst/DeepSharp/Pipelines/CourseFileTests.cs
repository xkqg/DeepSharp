// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A course saved as a file of its own: what has been said of each verb, in the order the steps belong. It is read through
/// the same door as a pipeline file and a preset — with the catalog of the verbs it may name, every fault at its line and
/// column, and a file from a newer version refused whole — because a second reader of the same steps would be a second set
/// of rules. It is a third kind of file under the one version the pipeline file has, so no file written before it changes
/// meaning and no reader before it is asked to read a version it does not know.
/// </summary>
public class CourseFileTests
{
    private static readonly StepCatalog Catalog = Shipped.Catalog();

    private static JsonObject Json(string text) => JsonNode.Parse(text)!.AsObject();

    private static IReadOnlyList<PipelineFileFault> Refused(string json, StepCatalog? catalog = null) =>
        Assert.Throws<PipelineFileException>(() => PipelineCourse.FromJson(json, catalog ?? Catalog)).Faults;

    private static IEnumerable<string> RootKeys(string json)
    {
        using var document = JsonDocument.Parse(json);

        return [.. document.RootElement.EnumerateObject().Select(property => property.Name)];
    }

    [Fact]
    public void ACourse_SurvivesTheFile_WithEverythingItHolds()
    {
        foreach (var course in new[] { PipelineCourse.Table, PipelineCourse.SeriesInTime, FilledCourses.Passengers(), FilledCourses.Prices() })
        {
            var read = PipelineCourse.FromJson(course.ToJson(), Catalog);

            Assert.Equal(course, read);
        }
    }

    [Fact]
    public void ACourseFilledIn_SurvivesTheFile_AsTheSamePipeline()
    {
        var filled = FilledCourses.Passengers();

        var read = PipelineCourse.FromJson(filled.ToJson(), Catalog);

        Assert.Equal(filled.ToDeclaration(Catalog), read.ToDeclaration(Catalog));
    }

    [Fact]
    public void ACourse_WritesItsVersionAndItsSteps_OneKeyToALine_WithALineFeedEveryWhere()
    {
        var course = PipelineCourse.Of([CourseStep.Of("settle.gaps", Json("""{"with":"zero"}"""))]);

        const string Expected = "{\n  \"version\": 8,\n  \"course\": [\n    {\n      \"step\": \"settle.gaps\",\n      \"with\": \"zero\"\n    }\n  ]\n}";

        Assert.Equal(Expected, course.ToJson());
        Assert.Equal(["version", "course"], RootKeys(course.ToJson()));
        Assert.DoesNotContain("\r", PipelineCourse.Table.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void AStepThatSaysNothing_IsWrittenAsItsVerbAlone()
    {
        var written = PipelineCourse.Of([CourseStep.Of("read.csv")]).ToJson();

        using var document = JsonDocument.Parse(written);
        var step = document.RootElement.GetProperty("course")[0];

        Assert.Equal(["step"], step.EnumerateObject().Select(property => property.Name));
    }

    [Fact]
    public void AKeyWrittenAsNothing_IsAKeyNobodyHasSaid()
    {
        const string File = """{"version": 7, "course": [{"step": "read.csv", "path": null}]}""";

        var read = PipelineCourse.FromJson(File, Catalog);

        Assert.Equal(PipelineCourse.Of([CourseStep.Of("read.csv")]), read);
        Assert.Equal(["path"], read.Steps[0].Waiting(Catalog));
    }

    [Theory]
    [InlineData("scale.given", """{"column":"year","lowest":1990}""")]
    [InlineData("scale.given", """{"highest":0}""")]
    [InlineData("feature.indicator", """{"indicator":"atr"}""")]
    [InlineData("outliers.clip", """{"bounds":"quantile"}""")]
    public void ACourseWrittenHalfWay_ReadsBackAsTheCourseItWas_WhateverTheValuesSaidAreTiedTo(string verb, string said)
    {
        // A value is judged beside the keys it is tied to, and some of those have not been said: 1990 for the lowest bound
        // beside the example 1 for the highest is wrong only because nobody has said the highest yet.
        var half = PipelineCourse.Of([CourseStep.Of(verb, Json(said))]);

        Assert.Equal(half, PipelineCourse.FromJson(half.ToJson(), Catalog));
    }

    [Fact]
    public void AStepThatSaysOnlyOneKeyItWaitsFor_IsNeverRefusedByTheFile_ForAnyVerb()
    {
        // Whatever the verb ties that key to, the others have not been said: the key said alone is the course the file keeps.
        foreach (var description in Catalog.Descriptions.Where(description => description.Waiting.Count > 0))
        {
            using var template = JsonDocument.Parse(description.Template);

            foreach (var key in description.Waiting)
            {
                var value = template.RootElement.GetProperty(key);
                JsonNode said = value.ValueKind == JsonValueKind.Number ? JsonValue.Create(1_000_000_000)! : JsonNode.Parse(value.GetRawText())!;
                var course = PipelineCourse.Of([CourseStep.Of(description.Verb, new JsonObject { [key] = said })]);

                Assert.Equal(course, PipelineCourse.FromJson(course.ToJson(), Catalog));
            }
        }
    }

    [Fact]
    public void AKeyTheVerbDoesNotTake_IsRefusedByTheFile_AndByTheDeclaration_InTheVerbsOwnWords()
    {
        var typo = PipelineCourse.Table.Say("read.csv", Json("""{"pth":"a.csv"}"""));

        var inTheFile = Refused(typo.ToJson());
        var inTheDeclaration = Assert.Throws<DeclarationException>(() => typo.ToDeclaration(Catalog)).Faults;

        Assert.Contains(inTheFile, fault => fault.Message.Contains("has no parameter called 'pth'", StringComparison.Ordinal));
        Assert.Contains(inTheDeclaration, fault => fault.Message.Contains("has no parameter called 'pth'", StringComparison.Ordinal));
    }

    [Fact]
    public void AFileThatNamesNoVersion_IsRefused_AsAPresetIs()
    {
        var faults = Refused("""{"course": [{"step": "read.csv"}]}""");

        var fault = Assert.Single(faults);
        Assert.Contains("names the version", fault.Message, StringComparison.Ordinal);
        Assert.StartsWith("A course file", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileWrittenAgainstANewerVersion_IsRefusedWhole_AndNothingInItIsRead()
    {
        // Whatever else is wrong with it is not said: it may hold words this reader never had.
        var faults = Refused("""{"version": 9, "course": "not a list", "surprise": true}""");

        var fault = Assert.Single(faults);
        Assert.Contains("version 9", fault.Message, StringComparison.Ordinal);
        Assert.Contains("newer DeepSharp", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKeyTheCourseFileDoesNotHold_IsRefused_NamingWhatItHolds()
    {
        var faults = Refused("""{"version": 7, "course": [{"step": "read.csv"}], "declaration": []}""");

        var fault = Assert.Single(faults);
        Assert.Contains("A course file has no 'declaration'", fault.Message, StringComparison.Ordinal);
        Assert.Contains("It holds: version, course", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineFile_IsNotACourse_AndTheRefusalSaysWhichKeyItLacks()
    {
        var pipeline = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Number("a")).Declaration.ToJson();

        var faults = Refused(pipeline);

        Assert.Contains(faults, fault => fault.Message.Contains("has no 'declaration'", StringComparison.Ordinal));
        Assert.Contains(faults, fault => fault.Message.Contains("under 'course'", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("""{"version": 7}""")]
    [InlineData("""{"version": 7, "course": "read.csv"}""")]
    [InlineData("""{"version": 7, "course": {"step": "read.csv"}}""")]
    public void AFileWithoutAListOfSteps_IsRefused(string json)
    {
        var faults = Refused(json);

        Assert.Contains(faults, fault => fault.Message.Contains("holds its steps as a list, under 'course'", StringComparison.Ordinal));
    }

    [Fact]
    public void ACourseWithNoStep_IsRefused()
    {
        var faults = Refused("""{"version": 7, "course": []}""");

        Assert.Contains(faults, fault => fault.Message.Contains("at least one step", StringComparison.Ordinal));
    }

    [Fact]
    public void AFileThatIsNotAnObject_IsRefused()
    {
        var faults = Refused("""["read.csv"]""");

        var fault = Assert.Single(faults);
        Assert.Contains("A course file is one JSON object", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryFault_IsAtItsStepsLineAndColumn_AndTheStepIsNumbered()
    {
        const string File = """
            {
              "version": 7,
              "course": [
                {"step": "read.csv"},
                {"step": "not.a.verb"},
                {"step": "scale.given", "column": "fare", "lowest": 5, "highest": 1},
                {"step": "settle.gaps", "wth": "zero"},
                "declare",
                {"path": "a.csv"}
              ]
            }
            """;

        var faults = Refused(File);

        Assert.Equal([5, 6, 7, 8, 9], faults.Select(fault => fault.Line));
        Assert.All(faults.Take(3), fault => Assert.Equal(5, fault.Column));
        Assert.Contains("Step 2", faults[0].Message, StringComparison.Ordinal);
        Assert.Contains("not a step anything here knows", faults[0].Message, StringComparison.Ordinal);
        Assert.Contains("Step 3", faults[1].Message, StringComparison.Ordinal);
        Assert.Contains("Step 4", faults[2].Message, StringComparison.Ordinal);
        Assert.Contains("'wth'", faults[2].Message, StringComparison.Ordinal);
        Assert.Contains("Step 5", faults[3].Message, StringComparison.Ordinal);
        Assert.Contains("an object", faults[3].Message, StringComparison.Ordinal);
        Assert.Contains("Step 6", faults[4].Message, StringComparison.Ordinal);
        Assert.Contains("names its verb", faults[4].Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AVerbAnotherPackageBrings_IsRefusedByACatalogNeverTaughtIt_NamingThePackage()
    {
        var faults = Refused(PipelineCourse.Table.ToJson(), StepCatalog.BuiltIn());

        var fault = Assert.Single(faults);
        Assert.Contains("DeepSharp.Learners.Networks", fault.Message, StringComparison.Ordinal);
        Assert.Contains("Step 12", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepWrittenAgainstAnOlderVersionThanItsVerb_IsRefusedByName()
    {
        // scale.given means what it says from version 7 of the file, so a file that says it was written against version 3
        // names a verb that did not exist then.
        var faults = Refused("""{"version": 3, "course": [{"step": "scale.given", "column": "fare", "lowest": 0, "highest": 512}]}""");

        var fault = Assert.Single(faults);
        Assert.Contains("version 7", fault.Message, StringComparison.Ordinal);
        Assert.Contains("written against version 3", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoStepsSaidInAnyOrderOfKeys_ReadAsTheSameCourse()
    {
        const string One = """{"version": 7, "course": [{"step": "split.byTime", "gap": 1, "column": "Date"}]}""";
        const string Other = """{"version": 7, "course": [{"column": "Date", "step": "split.byTime", "gap": 1}]}""";

        Assert.Equal(PipelineCourse.FromJson(One, Catalog), PipelineCourse.FromJson(Other, Catalog));
    }

    [Fact]
    public void ReadingAndWriting_NeedACatalogAndAFile()
    {
        Assert.Throws<ArgumentNullException>(() => PipelineCourse.FromJson(null!, Catalog));
        Assert.Throws<ArgumentNullException>(() => PipelineCourse.FromJson("{}", null!));
    }
}
