// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The form face of a block: Verso's properties panel, built from the step's own description. Every field is read
/// from the step as it writes itself, and every change is written into the step's text and read back through the
/// catalog a pipeline file is read with — so the text and the form are two faces of one step, and a value the step
/// refuses is never written. The form writes the block's text, never the cell's metadata; it never throws, since
/// Verso shows an empty panel for a part that does; and it writes a number the way a pipeline file does, whatever
/// language the interface speaks.
/// </summary>
public sealed class FormTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-form-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public FormTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static StepForm Form(Notebook notebook) => notebook.Host.GetPropertyProviders().OfType<StepForm>().Single();

    private static Task<PropertySection> SectionAsync(Notebook notebook, CellModel cell) =>
        Form(notebook).GetPropertiesSectionAsync(cell, new RenderGesture(notebook, cell));

    private static Task ChangeAsync(Notebook notebook, CellModel cell, string field, object? value) =>
        Form(notebook).OnPropertyChangedAsync(cell, field, value, new RenderGesture(notebook, cell));

    private static PropertyField Field(PropertySection section, string name) => section.Fields.Single(field => field.Name == name);

    private static IPipelineStep Step(CellModel cell) => StepCatalog.BuiltIn().WithIndicators().ReadStep(cell.Source);

    [Fact]
    public async Task TheFormIsAPartVersoLoads_ForBlocksAlone()
    {
        await using var notebook = await NotebookAsync(Titanic[0]);
        var code = notebook.Scaffold.AddCell("code", source: "1 + 1");

        Assert.Equal(StepForm.Id, Form(notebook).ExtensionId);
        Assert.True(Form(notebook).AppliesTo(notebook.Scaffold.Cells[0], null!));
        Assert.False(Form(notebook).AppliesTo(code, null!));
        Assert.False(string.IsNullOrWhiteSpace(Form(notebook).Name));
        Assert.False(string.IsNullOrWhiteSpace(Form(notebook).Description));
        Assert.Equal(0, Form(notebook).Order);
    }

    [Fact]
    public async Task TheTextAndTheFormDescribeTheSameStep_ForEveryVerb()
    {
        // Every verb's starter block: a field for every key the step takes, the verb chosen among the verbs of its
        // stage, and every value read as the step holds it — so writing any field back as it stands changes nothing.
        var catalog = NotebookVerbs.Catalog();
        await using var notebook = await NotebookAsync();

        foreach (var description in catalog.Descriptions)
        {
            var cell = notebook.AddBlock(catalog.ReadStep(description.Template).AsBlockText());
            var before = cell.Source;
            var section = await SectionAsync(notebook, cell);
            var verb = Field(section, "step");

            Assert.Equal(PropertyFieldType.Select, verb.FieldType);
            Assert.Equal(description.Verb, verb.CurrentValue);
            Assert.Contains(verb.Options!, option => option.Value == description.Verb);
            Assert.Equal(description.Purpose, section.Description);

            foreach (var key in description.Parameters.SelectMany(parameter => parameter.Keys))
            {
                Assert.Contains(section.Fields, field => field.Name == key || field.Name.StartsWith($"{key}/", StringComparison.Ordinal));
            }

            foreach (var field in section.Fields.Where(field => !field.IsReadOnly))
            {
                await ChangeAsync(notebook, cell, field.Name, field.CurrentValue);

                Assert.True(before == cell.Source, $"{description.Verb}.{field.Name}: {cell.Source}");
                Assert.DoesNotContain("not made", (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);

                // A field the form draws is one it takes back: some kind of the step claims it.
                if (field.Name != "step")
                {
                    var json = System.Text.Json.Nodes.JsonNode.Parse(cell.Source)!.AsObject();
                    var edit = new FormEdit(field.Name, FieldValue.Of(field.CurrentValue), json, FormScope.Unknown, Step(cell));

                    Assert.True(description.Parameters.Any(parameter => parameter.Accept(edit)), $"{description.Verb}.{field.Name} is claimed by no kind");
                }
            }
        }
    }

    [Fact]
    public async Task TheVerbField_OffersOnlyTheVerbsThatActAsItsStepActs()
    {
        // Leaving a column out and leaving rows out are both cleaning, and do different things: a block below the
        // split may leave a column out and may not leave rows out, so the one is never offered in place of the other.
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "drop.columns", "columns": ["age_was_missing"]}"""]);

        var verbs = Field(await SectionAsync(notebook, notebook.Scaffold.Cells[5]), "step").Options!.Select(option => option.Value);

        Assert.Equal(["drop.columns"], verbs);
    }

    [Fact]
    public void EveryFieldNameTheFormBuilds_ReadsBackAsTheKeyAndThePartItNames()
    {
        Assert.True(FormVocabulary.IsKind(FormVocabulary.Kind("columns", "a/b"), "columns", out var kind) && kind == "a/b");
        Assert.True(FormVocabulary.IsAbsent(FormVocabulary.Absent("columns", "x"), "columns", out var absent) && absent == "x");
        Assert.True(FormVocabulary.IsMember(FormVocabulary.Member("columns", "age"), "columns", out var member) && member == "age");
        Assert.True(FormVocabulary.IsPlace(FormVocabulary.Place("columns", 2), "columns", out var place) && place == 2);
        Assert.Equal("with/value", FormVocabulary.StrategyValue("with"));

        Assert.False(FormVocabulary.IsKind("columns/optional/x", "columns", out _));
        Assert.False(FormVocabulary.IsAbsent("columns/kind/x", "columns", out _));
        Assert.False(FormVocabulary.IsMember("parts/x", "columns", out _));
        Assert.False(FormVocabulary.IsPlace("columns/first", "columns", out _));
        Assert.False(FormVocabulary.IsPlace("columns/-1", "columns", out _));
        Assert.False(FormVocabulary.IsPlace("period", "columns", out _));
    }

    [Fact]
    public async Task AFormChange_RewritesTheBlocksTextAsTheStepWritesItself_AndNeverItsMetadata()
    {
        await using var notebook = await NotebookAsync(Titanic[4]);
        var cell = notebook.Scaffold.Cells[0];

        await ChangeAsync(notebook, cell, "scale", "minmax");

        Assert.Equal(new NormaliseStep("fare", Scale.MinMax, OutOfRange.Pass).AsBlockText(), cell.Source);
        Assert.Empty(cell.Metadata);
    }

    [Fact]
    public async Task AFormValueTheStepRefuses_LeavesTheTextAsItWas_AndSaysWhy_UntilTheBlockChanges()
    {
        await using var notebook = await NotebookAsync("""{"step": "split.atRandom", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 1}""");
        var cell = notebook.Scaffold.Cells[0];
        var before = cell.Source;

        // Nine tenths to train leaves less than nothing to test.
        await ChangeAsync(notebook, cell, "train", "0.9");

        Assert.Equal(before, cell.Source);
        Assert.Contains("not made", (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);

        cell.Source = """{"step": "split.atRandom", "train": 0.6, "validation": 0.2, "test": 0.2, "seed": 1}""";

        Assert.DoesNotContain("not made", (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AFormOverTextThatIsNotAStepYet_SaysWhy_InsteadOfShowingNothing()
    {
        await using var notebook = await NotebookAsync("""{"step": "normalise", "column": """);
        var cell = notebook.Scaffold.Cells[0];
        var before = cell.Source;

        var section = await SectionAsync(notebook, cell);
        await ChangeAsync(notebook, cell, "scale", "minmax");

        Assert.Empty(section.Fields);
        Assert.Contains("(1,", section.Description, StringComparison.Ordinal);
        Assert.Equal(before, cell.Source);
    }

    [Fact]
    public async Task AColumnFieldWithoutAKnownScope_IsTextNotAnEmptyList()
    {
        await using var notebook = await NotebookAsync(Titanic[4], """{"step": "drop.columns", "columns": ["a", "b"]}""");

        var column = Field(await SectionAsync(notebook, notebook.Scaffold.Cells[0]), "column");
        var columns = Field(await SectionAsync(notebook, notebook.Scaffold.Cells[1]), "columns");

        Assert.Equal(PropertyFieldType.Text, column.FieldType);
        Assert.Equal("fare", column.CurrentValue);
        Assert.Equal(PropertyFieldType.Text, columns.FieldType);
        Assert.Equal("""["a", "b"]""", columns.CurrentValue);

        await ChangeAsync(notebook, notebook.Scaffold.Cells[1], "columns", """["a", "c"]""");

        Assert.Equal(["a", "c"], ((DropColumnsStep)Step(notebook.Scaffold.Cells[1])).Columns);
    }

    [Fact]
    public async Task OnceAGestureShowedThePipeline_AColumnIsPickedFromTheColumnsInScopeAtItsBlock_OfAKindItTakes()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "drop.columns", "columns": ["age_was_missing"]}"""]);
        var normalise = notebook.Scaffold.Cells[4];
        var drop = notebook.Scaffold.Cells[5];

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        var column = Field(await SectionAsync(notebook, normalise), "column");

        Assert.Equal(PropertyFieldType.Select, column.FieldType);
        Assert.Equal(["survived", "pclass", "age", "fare", "age_was_missing"], column.Options!.Select(option => option.Value));
        Assert.Equal("fare", column.CurrentValue);

        // A list of columns is one switch per column in scope, each on when the list names it.
        var toggles = (await SectionAsync(notebook, drop)).Fields.Where(field => field.Name.StartsWith("columns/", StringComparison.Ordinal)).ToArray();

        Assert.All(toggles, toggle => Assert.Equal(PropertyFieldType.Toggle, toggle.FieldType));
        Assert.Equal(["survived", "pclass", "age", "fare", "age_was_missing"], toggles.Select(toggle => toggle.DisplayName));
        Assert.Equal([false, false, false, false, true], toggles.Select(toggle => (bool)toggle.CurrentValue!));

        await ChangeAsync(notebook, drop, "columns/age", true);
        await ChangeAsync(notebook, drop, "columns/age_was_missing", false);

        Assert.Equal(["age"], ((DropColumnsStep)Step(drop)).Columns);
    }

    [Theory]
    [InlineData("")]
    [InlineData("nl-NL")]
    public async Task ANumberIsReadAndWrittenAsAPipelineFileWritesIt_WhateverLanguageTheInterfaceSpeaks(string culture)
    {
        var before = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(culture);
            await using var notebook = await NotebookAsync("""{"step": "split.atRandom", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""");
            var cell = notebook.Scaffold.Cells[0];
            var section = await SectionAsync(notebook, cell);

            Assert.Equal("0.7", Field(section, "train").CurrentValue);
            Assert.Equal("20260923", Field(section, "seed").CurrentValue);
            Assert.True(Field(section, "test").IsReadOnly);

            // The test share is what the others leave.
            await ChangeAsync(notebook, cell, "train", "0.6");

            var split = (SplitAtRandomStep)Step(cell);

            Assert.Equal(new SplitShares(0.6, 0.15, 0.25), split.Shares);
            Assert.Equal("0.25", Field(await SectionAsync(notebook, cell), "test").CurrentValue);

            // A decimal comma is not guessed at: the form says how a pipeline writes the number.
            var written = cell.Source;
            await ChangeAsync(notebook, cell, "validation", "0,2");

            Assert.Equal(written, cell.Source);
            Assert.Contains("0.2", (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }

    [Fact]
    public async Task EveryKindHasItsControl_AndEachWritesTheValueItsKindHolds()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "feature.timeParts", "column": "when", "parts": ["hour"], "asCategories": false}""",
            """{"step": "fill.missing", "column": "age", "with": "median"}""",
            """{"step": "feature.add", "column": "total", "left": "a", "arithmetic": "plus", "right": "b"}""");
        var parts = notebook.Scaffold.Cells[0];
        var fill = notebook.Scaffold.Cells[1];

        var section = await SectionAsync(notebook, parts);

        Assert.Equal(PropertyFieldType.MultiSelect, Field(section, "parts").FieldType);
        Assert.Equal("hour", Field(section, "parts").CurrentValue);
        Assert.Equal(PropertyFieldType.Toggle, Field(section, "asCategories").FieldType);

        // Verso hands a choice of several back joined with commas, or as a list: both are read.
        await ChangeAsync(notebook, parts, "parts", "hour,dayofweek");
        Assert.Equal([TimePart.Hour, TimePart.DayOfWeek], ((TimePartsStep)Step(parts)).Parts);

        await ChangeAsync(notebook, parts, "parts", new[] { "month" });
        await ChangeAsync(notebook, parts, "asCategories", true);
        Assert.Equal([TimePart.Month], ((TimePartsStep)Step(parts)).Parts);
        Assert.True(((TimePartsStep)Step(parts)).AsCategories);

        // A way to fill that takes a number grows a field for it, which starts at nought.
        Assert.DoesNotContain((await SectionAsync(notebook, fill)).Fields, field => field.Name == "with/value");

        await ChangeAsync(notebook, fill, "with", "constant");
        Assert.Equal("0", Field(await SectionAsync(notebook, fill), "with/value").CurrentValue);

        await ChangeAsync(notebook, fill, "with/value", "2.5");
        Assert.Equal(With.Constant(2.5), ((FillMissingStep)Step(fill)).Strategy);

        // A share that may be left out is left out by clearing it.
        await ChangeAsync(notebook, fill, "refuseAbove", "0.5");
        Assert.Equal(0.5, ((FillMissingStep)Step(fill)).RefuseAbove);

        await ChangeAsync(notebook, fill, "refuseAbove", "");
        Assert.Null(((FillMissingStep)Step(fill)).RefuseAbove);
    }

    [Fact]
    public async Task AWholeNumberAFileMayLeaveOut_ShowsWhatLeavingItOutMeans_AndIsLeftOutByClearingIt()
    {
        // A split in time writes no gap until it has one, so the texts written before it had the parameter stay as
        // they were; the form still shows it, as the nought leaving it out means.
        await using var notebook = await NotebookAsync(
            """{"step": "split.byTime", "column": "when", "train": 0.7, "validation": 0.15, "test": 0.15, "predict": 0}""");
        var split = notebook.Scaffold.Cells[0];

        Assert.Equal("0", Field(await SectionAsync(notebook, split), "gap").CurrentValue);

        await ChangeAsync(notebook, split, "gap", "3");
        Assert.Equal(3, ((SplitByTimeStep)Step(split)).Gap);
        Assert.Contains("\"gap\": 3", split.Source, StringComparison.Ordinal);

        await ChangeAsync(notebook, split, "gap", "");
        Assert.Equal(0, ((SplitByTimeStep)Step(split)).Gap);
        Assert.DoesNotContain("gap", split.Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AColumnAStepCanDoWithout_IsLeftOutByClearingIt()
    {
        // What a distribution's shares are shares of may be left out, and clearing it leaves it out: the text then
        // says nothing about it, as a text that never had one does.
        await using var notebook = await NotebookAsync(
            """{"step": "target.distribution", "columns": ["w500", "w550"], "scaleBy": "chicks"}""");
        var output = notebook.Scaffold.Cells[0];

        Assert.Equal("chicks", Field(await SectionAsync(notebook, output), "scaleBy").CurrentValue);

        await ChangeAsync(notebook, output, "scaleBy", "");
        Assert.Null(((DistributionStep)Step(output)).ScaleBy);
        Assert.DoesNotContain("scaleBy", output.Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheVerb_IsSwitchedWithinItsStage_KeepingEveryValueTheOtherVerbTakes()
    {
        await using var notebook = await NotebookAsync("""{"step": "split.atRandom", "train": 0.6, "validation": 0.2, "test": 0.2, "seed": 7}""");
        var cell = notebook.Scaffold.Cells[0];

        var verbs = Field(await SectionAsync(notebook, cell), "step").Options!.Select(option => option.Value).ToArray();

        Assert.All(verbs, verb => Assert.StartsWith("split.", verb, StringComparison.Ordinal));
        Assert.Contains("split.stratified", verbs);

        await ChangeAsync(notebook, cell, "step", "split.stratified");

        var stratified = (SplitStratifiedStep)Step(cell);

        Assert.Equal(new SplitShares(0.6, 0.2, 0.2), stratified.Shares);
        Assert.Equal(7, stratified.Seed);
    }

    [Fact]
    public async Task TheSchemasForm_OffersEveryColumnTheSourceHas_TakenOrNot_InTheSourcesOrder()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        // Before anything read the source, the form knows the declared columns alone.
        Assert.Equal(4, (await SectionAsync(notebook, declare)).Fields.Count(field => field.Name.StartsWith("columns/kind/", StringComparison.Ordinal)));

        await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Show);

        var section = await SectionAsync(notebook, declare);
        var kinds = section.Fields.Where(field => field.Name.StartsWith("columns/kind/", StringComparison.Ordinal)).ToArray();

        Assert.Equal(15, kinds.Length);
        Assert.Equal("integer", Field(section, "columns/kind/pclass").CurrentValue);
        Assert.Equal(FormVocabulary.NotTaken, Field(section, "columns/kind/sex").CurrentValue);
        Assert.True((bool)Field(section, "columns/optional/age").CurrentValue!);
        Assert.DoesNotContain(section.Fields, field => field.Name == "columns/optional/sex");

        await ChangeAsync(notebook, declare, "columns/kind/sex", "category");
        await ChangeAsync(notebook, declare, "columns/kind/pclass", FormVocabulary.NotTaken);
        await ChangeAsync(notebook, declare, "columns/optional/fare", true);

        var schema = (DeclareStep)Step(declare);

        // A column not taken stays in the schema, excluded with its kind, so taking it in again brings it back as it was.
        Assert.Equal(["survived", "sex", "age", "fare"], schema.Taking.Select(column => column.Name));
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Integer, Optional: false) { Excluded = true }, schema.Columns[1]);
        Assert.Equal(ColumnKind.Category, schema.Taking.Single(column => column.Name == "sex").Kind);
        Assert.True(schema.Taking.Single(column => column.Name == "fare").Optional);
        Assert.Equal(FormVocabulary.NotTaken, Field(await SectionAsync(notebook, declare), "columns/kind/pclass").CurrentValue);

        // A column the source does not have goes last.
        await ChangeAsync(notebook, declare, "columns/kind/extra", "number");

        Assert.Equal("extra", ((DeclareStep)Step(declare)).Columns[^1].Name);
    }

    [Fact]
    public async Task TheSchemasForm_ShowsAnExcludedColumnAsNotTaken_AndAKindPickTakesItInAgainWithThatKind()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0],
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false, "excluded": true}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
            Titanic[2],
            Titanic[3],
            Titanic[4]);
        var declare = notebook.Scaffold.Cells[1];
        var section = await SectionAsync(notebook, declare);

        Assert.Equal(FormVocabulary.NotTaken, Field(section, "columns/kind/pclass").CurrentValue);
        Assert.DoesNotContain(section.Fields, field => field.Name == "columns/optional/pclass");

        await ChangeAsync(notebook, declare, "columns/kind/pclass", "number");

        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Number, Optional: false), ((DeclareStep)Step(declare)).Columns[1]);
    }

    [Fact]
    public async Task NotTakenOnAColumnAStepReads_ExcludesIt_AndTheStepThatReadsItSaysSo()
    {
        // The form changes the schema alone: the step below that reads the column is not rewritten, and says that what
        // it reads is no longer there.
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        await ChangeAsync(notebook, declare, "columns/kind/age", FormVocabulary.NotTaken);

        Assert.True(((DeclareStep)Step(declare)).Columns.Single(column => column.Name == "age").Excluded);
        Assert.Contains(
            NotebookPipeline.Of(notebook.Scaffold.Cells).Blocks[3].Faults,
            fault => fault.Contains("which the schema excludes", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AKindPickOnACategoryThatSaysWhatItWas_GivesItThatKind_AndForgetsWhatItWas()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0],
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "category", "optional": false, "was": "integer"}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
            Titanic[2],
            Titanic[3],
            Titanic[4]);
        var declare = notebook.Scaffold.Cells[1];

        await ChangeAsync(notebook, declare, "columns/kind/pclass", "number");

        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Number, Optional: false), ((DeclareStep)Step(declare)).Columns[1]);
        Assert.DoesNotContain("not made", (await SectionAsync(notebook, declare)).Description, StringComparison.Ordinal);
    }

    [Fact]
    public async Task NotTakenOnTheOnlyColumnASchemaTakes_IsRefused_InTheSchemasOwnWords()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0], """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""");
        var declare = notebook.Scaffold.Cells[1];
        var before = declare.Source;

        await ChangeAsync(notebook, declare, "columns/kind/survived", FormVocabulary.NotTaken);

        var description = (await SectionAsync(notebook, declare)).Description;

        Assert.Equal(before, declare.Source);
        Assert.Contains("so no column would take part.", description, StringComparison.Ordinal);
        Assert.DoesNotContain("(Parameter", description, StringComparison.Ordinal);
    }

    [Fact]
    public async Task WhetherAColumnNotTakenMayBeAbsent_SaysNothing_EvenWhenTheSchemaNamesIt()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0],
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false, "excluded": true}]}""");
        var declare = notebook.Scaffold.Cells[1];
        var before = declare.Source;

        await ChangeAsync(notebook, declare, "columns/optional/pclass", true);

        Assert.Equal(before, declare.Source);
        Assert.Contains("is not taken", (await SectionAsync(notebook, declare)).Description, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AValueAFrontEndSendsAsJson_OrAsText_IsReadAsTheValueItHolds()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "feature.timeParts", "column": "when", "parts": ["hour"], "asCategories": false}""",
            """{"step": "fill.missing", "column": "age", "with": "median"}""");
        var parts = notebook.Scaffold.Cells[0];
        var fill = notebook.Scaffold.Cells[1];

        static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement;

        await ChangeAsync(notebook, parts, "parts", Json("""["month", "quarter"]"""));
        await ChangeAsync(notebook, parts, "asCategories", Json("true"));

        Assert.Equal([TimePart.Month, TimePart.Quarter], ((TimePartsStep)Step(parts)).Parts);
        Assert.True(((TimePartsStep)Step(parts)).AsCategories);

        await ChangeAsync(notebook, parts, "asCategories", Json("false"));
        await ChangeAsync(notebook, parts, "column", Json("\"stamp\""));

        Assert.False(((TimePartsStep)Step(parts)).AsCategories);
        Assert.Equal("stamp", ((TimePartsStep)Step(parts)).Column);

        // A switch a front end hands back as the words true or false.
        await ChangeAsync(notebook, parts, "asCategories", "true");

        Assert.True(((TimePartsStep)Step(parts)).AsCategories);

        await ChangeAsync(notebook, fill, "refuseAbove", Json("0.25"));

        Assert.Equal(0.25, ((FillMissingStep)Step(fill)).RefuseAbove);
        Assert.Equal("0.25", Field(await SectionAsync(notebook, fill), "refuseAbove").CurrentValue);

        await ChangeAsync(notebook, fill, "refuseAbove", Json("null"));

        Assert.Null(((FillMissingStep)Step(fill)).RefuseAbove);
    }

    [Fact]
    public async Task AValueNotOfItsKind_IsRefused_InWordsTheFormShows()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "feature.timeParts", "column": "when", "parts": ["hour"], "asCategories": false}""",
            """{"step": "drop.columns", "columns": ["a"]}""",
            """{"step": "outliers.clip", "column": "x", "bounds": "iqr", "at": 1.5, "outlier": "clip"}""",
            """{"step": "split.atRandom", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 1}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}]}""");

        async Task SaysAsync(int block, string field, object? value, string words)
        {
            var cell = notebook.Scaffold.Cells[block];
            var before = cell.Source;

            await ChangeAsync(notebook, cell, field, value);

            Assert.Equal(before, cell.Source);
            Assert.Contains(words, (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);
        }

        await SaysAsync(0, "asCategories", "maybe", "neither true nor false");
        await SaysAsync(0, "parts", null, "not made");
        await SaysAsync(1, "columns", "a, b", "written as JSON");
        await SaysAsync(1, "columns", "null", "not made");
        await SaysAsync(2, "at", "abc", "is not a number");
        await SaysAsync(2, "at", "NaN", "is not a number");
        await SaysAsync(3, "train", "abc", "is not a share");
        await SaysAsync(4, "columns/optional/b", true, "is not taken");
    }

    [Fact]
    public async Task ANumberBeyondWhatADecimalHolds_IsStillWrittenAsTheNumberItIs()
    {
        await using var notebook = await NotebookAsync("""{"step": "outliers.clip", "column": "x", "bounds": "iqr", "at": 1.5, "outlier": "clip"}""");
        var cell = notebook.Scaffold.Cells[0];

        await ChangeAsync(notebook, cell, "at", "1e30");

        Assert.Equal(1e30, ((ClipOutliersStep)Step(cell)).At);
    }

    [Fact]
    public async Task AFieldNoKindOfTheStepOwns_ChangesNothing()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "fill.missing", "column": "age", "with": "median"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}]}""",
            """{"step": "feature.indicator", "column": "range", "indicator": "atr", "columns": ["high", "low", "close"], "period": 14}""");

        async Task NothingAsync(int block, string field, object? value)
        {
            var cell = notebook.Scaffold.Cells[block];
            var before = cell.Source;

            await ChangeAsync(notebook, cell, field, value);

            Assert.Equal(before, cell.Source);
            Assert.DoesNotContain("not made", (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);
        }

        // A way of filling that carries no number has no number to set; a declared column's field is its kind or
        // whether it may be absent; an indicator has as many places as it reads.
        await NothingAsync(0, "with/value", "2");
        await NothingAsync(1, "columns/colour", "red");
        await NothingAsync(2, "columns/9", "close");
        await NothingAsync(2, "columns/first", "close");
    }

    [Fact]
    public async Task AListThatMayBeLeftOut_IsLeftOut_OnceNoColumnIsOn()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "evidence.profile"}"""]);
        var profile = notebook.Scaffold.Cells[5];

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.All(
            (await SectionAsync(notebook, profile)).Fields.Where(field => field.Name.StartsWith("columns/", StringComparison.Ordinal)),
            toggle => Assert.False((bool)toggle.CurrentValue!));

        await ChangeAsync(notebook, profile, "columns/age", true);

        Assert.Equal(["age"], ((ProfileStep)Step(profile)).Columns);

        await ChangeAsync(notebook, profile, "columns/age", false);

        Assert.Empty(((ProfileStep)Step(profile)).Columns);
        Assert.DoesNotContain("columns", profile.Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AColumnAStepHoldsBeyondTheKnownColumns_IsStillAmongItsPicks()
    {
        // A schema that keeps the rest lets a step read a column the schema never named; its pick still shows it.
        await using var notebook = await NotebookAsync(
            Titanic[0],
            """{"step": "declare", "remainder": "keep", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""",
            """{"step": "maths", "column": "sibsp", "maths": "log1p", "into": "sibsp"}""");

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        var column = Field(await SectionAsync(notebook, notebook.Scaffold.Cells[2]), "column");

        Assert.Equal(PropertyFieldType.Select, column.FieldType);
        Assert.Equal(["survived", "sibsp"], column.Options!.Select(option => option.Value));
    }

    [Fact]
    public async Task WhereTheLastGestureReadNoRows_OrNeverReadTheBlock_TheFormPicksNothing()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "read.csv", "path": "missing.csv"}""",
            Titanic[1],
            """{"step": "split.stratified", "column": "survived"}""",
            Titanic[4]);
        var read = notebook.Scaffold.Cells[0];
        var declare = notebook.Scaffold.Cells[1];

        async Task<int> KindsAsync() =>
            (await SectionAsync(notebook, declare)).Fields.Count(field => field.Name.StartsWith("columns/kind/", StringComparison.Ordinal));

        // The rows were refused: no source is known, only what the schema declares.
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(4, await KindsAsync());

        // Below a block that is not a step, and a block added since the gesture, are in no pipeline it read.
        Assert.Equal(PropertyFieldType.Text, Field(await SectionAsync(notebook, notebook.Scaffold.Cells[3]), "column").FieldType);
        Assert.Equal(PropertyFieldType.Text, Field(await SectionAsync(notebook, notebook.AddBlock(Titanic[4])), "column").FieldType);

        // Rows read, then the source moved to a file that is not there: the rows kept are another source's.
        read.Source = Titanic[0];
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(15, await KindsAsync());

        read.Source = """{"step": "read.csv", "path": "missing.csv"}""";
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(4, await KindsAsync());
    }

    [Fact]
    public async Task WhereTheLastGestureReadNoSourceFile_TheSchemaKnowsItsOwnColumnsAlone()
    {
        // A first block that is not a step makes an empty pipeline; rows handed in have no file to read.
        await using var notebook = await NotebookAsync("""{"step": "read.csv"}""", Titanic[4]);
        var normalise = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(normalise, StepRenderer.Show);

        Assert.Equal(PropertyFieldType.Text, Field(await SectionAsync(notebook, normalise), "column").FieldType);

        await using var handed = await NotebookAsync("""{"step": "read.rows", "description": "rows handed in"}""", Titanic[1]);
        var declare = handed.Scaffold.Cells[1];

        await handed.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(4, (await SectionAsync(handed, declare)).Fields.Count(field => field.Name.StartsWith("columns/kind/", StringComparison.Ordinal)));
    }

    [Fact]
    public async Task AFormVersoNeverLoaded_ShowsTheStepButChangesNothing_WhileOneWhoseBlockTypeIsOff_StillKnowsTheScope()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];
        var before = normalise.Source;

        await notebook.GestureAsync(normalise, StepRenderer.Show);

        // A form with no notebook knows no columns, and has no notebook to change a block of.
        var alone = new StepForm();

        Assert.Equal(PropertyFieldType.Text, Field(await alone.GetPropertiesSectionAsync(normalise, new RenderGesture(notebook, normalise)), "column").FieldType);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(
            () => alone.OnPropertyChangedAsync(normalise, "scale", "minmax", new RenderGesture(notebook, normalise)));

        Assert.Contains("not loaded by Verso", refused.Message, StringComparison.Ordinal);
        Assert.Equal(before, normalise.Source);

        // Switched off, the block type still keeps the notebook's one session, and the form still reads it.
        await notebook.Host.DisableExtensionAsync(StepCellType.Id);

        Assert.Equal(PropertyFieldType.Select, Field(await SectionAsync(notebook, normalise), "column").FieldType);
    }

    [Fact]
    public async Task AnIndicatorsColumns_AreOnePickPerPlace_EachARoleOfItsOwn()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "read.csv", "path": "prices.csv"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "t", "kind": "integer", "optional": false}, {"name": "high", "kind": "number", "optional": false}, {"name": "low", "kind": "number", "optional": false}, {"name": "close", "kind": "number", "optional": false}]}""",
            """{"step": "order.by", "columns": ["t"]}""",
            """{"step": "feature.indicator", "column": "range", "indicator": "atr", "columns": ["high", "low", "close"], "period": 14}""");
        File.WriteAllText(Path.Join(_folder, "prices.csv"), "t,high,low,close\n1,2,1,1.5\n2,3,2,2.5\n");
        var indicator = notebook.Scaffold.Cells[3];

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        var section = await SectionAsync(notebook, indicator);

        Assert.Equal(["high", "low", "close"], new[] { "columns/0", "columns/1", "columns/2" }.Select(name => Field(section, name).CurrentValue));
        Assert.All(new[] { "columns/0", "columns/1", "columns/2" }, name => Assert.Equal(PropertyFieldType.Select, Field(section, name).FieldType));

        await ChangeAsync(notebook, indicator, "columns/0", "close");
        await ChangeAsync(notebook, indicator, "columns/1", "close");

        Assert.Equal(["close", "close", "close"], ((AddIndicatorStep)Step(indicator)).Columns);
    }

    [Fact]
    public async Task AFormChange_ClearsTheBlock_AndEveryViewItMadeStale_AndWithdrawsTheHandOver()
    {
        // The fill stands below the split, so the rows at the source are not worked out from it; the normalise
        // below it is, and so is the fill's own card.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];
        var fill = notebook.Scaffold.Cells[3];
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(read, StepRenderer.Show);
        await notebook.GestureAsync(normalise, StepRenderer.Show);
        await notebook.GestureAsync(fill, StepRenderer.Show);

        Assert.True(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));

        await ChangeAsync(notebook, fill, "with", "mean");

        Assert.Equal(2, read.Outputs.Count);
        Assert.Empty(fill.Outputs);
        Assert.Empty(normalise.Outputs);
        Assert.False(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));
    }

    [Fact]
    public async Task AViewOfABlockDeletedSince_IsForgotten_WhenAFormChangeClearsStaleViews()
    {
        // Shown, deleted, and then the notebook read again by a gesture elsewhere: the view's block is in no
        // pipeline any more, and a change made in a form forgets it rather than failing to find it.
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];
        var fill = notebook.Scaffold.Cells[3];

        await notebook.GestureAsync(normalise, StepRenderer.Show);
        notebook.Scaffold.RemoveCell(normalise.Id);
        await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Show);

        await ChangeAsync(notebook, fill, "with", "mean");

        Assert.Equal(With.Mean, ((FillMissingStep)Step(fill)).Strategy);
        Assert.DoesNotContain(normalise.Id, notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session.Shown.Keys);
    }

    [Fact]
    public async Task AFormEdit_WithdrawsThePipelineFromTheVariablesVersoHandsIt_GestureOrNot()
    {
        // A C# cell may have been handed the pipeline before this form was ever used: the form takes it back from
        // the notebook's own variables, the ones Verso hands it, with no gesture needed first.
        await using var notebook = await NotebookAsync(Titanic);
        var declaration = new PipelineDeclaration([.. Titanic.Select(block => StepCatalog.BuiltIn().ReadStep(block))]);

        notebook.Scaffold.Variables.Set(StepKernel.HandOver, declaration.ToJson());

        await ChangeAsync(notebook, notebook.Scaffold.Cells[3], "with", "mean");

        Assert.False(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));
    }

    [Fact]
    public async Task WhatNoFieldOfTheStepIs_ChangesNothing_AndAValueNotOfItsKindIsRefused_NeitherThrowing()
    {
        await using var notebook = await NotebookAsync(Titanic[4]);
        var cell = notebook.Scaffold.Cells[0];
        var before = cell.Source;

        await ChangeAsync(notebook, cell, "colour", "red");

        Assert.DoesNotContain("not made", (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);

        async Task RefusedAsync(string field, object? value)
        {
            await ChangeAsync(notebook, cell, field, value);

            Assert.Equal(before, cell.Source);
            Assert.Contains("not made", (await SectionAsync(notebook, cell)).Description, StringComparison.Ordinal);
        }

        await RefusedAsync("scale", 42);
        await RefusedAsync("scale", null);
        await RefusedAsync("scale", new object());
        await RefusedAsync("step", "no.such.verb");
        await RefusedAsync("column", " ");
    }
}
