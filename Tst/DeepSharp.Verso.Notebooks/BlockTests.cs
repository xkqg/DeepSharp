// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using Verso.Abstractions;
using Verso.Extensions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// One block of a notebook is one step of the pipeline, written as the step's own JSON. Running a block reads it
/// through the catalog a pipeline file is read with, says which stage of the pipeline it belongs to, and puts
/// each fault at its line and column; an editor is offered the verbs, the keys and the words a step takes while
/// it is typed, and told what each one means.
/// </summary>
public class BlockTests
{
    private static StepCatalog Verbs() => StepCatalog.BuiltIn().WithIndicators();

    private static IReadOnlyList<Type> Parts() =>
        [.. typeof(StepCellType).Assembly.GetTypes().Where(type => type.GetCustomAttribute<VersoExtensionAttribute>() is not null)];

    [Fact]
    public void ABlockIsACellTypeOfItsOwn_EditedAsText_WhoseOutputIsNeverSaved()
    {
        // What a block shows is worked out from the data each time it is asked for, and the data is somebody's:
        // a notebook file carries the pipeline, never the rows.
        var cell = new StepCellType();

        Assert.Equal("deepsharp.step", StepCellType.StepType);
        Assert.Equal(StepCellType.StepType, cell.CellTypeId);
        Assert.True(cell.IsEditable);
        Assert.False(cell.PersistsOutputs);
        Assert.Equal("pdd", StepKernel.Language);
        Assert.Equal(StepKernel.Language, cell.Kernel.LanguageId);
        Assert.Equal(cell.CellTypeId, cell.Renderer.CellTypeId);
        Assert.False(string.IsNullOrWhiteSpace(cell.DisplayName));
        Assert.False(string.IsNullOrWhiteSpace(cell.Icon));
        Assert.Equal("read.csv", Verbs().ReadStep(cell.GetDefaultContent()).Verb);
    }

    [Fact]
    public void ABlocksText_HasOneLineEnding_WhateverMachineOrRuntimeWritesIt()
    {
        // A block is saved in the notebook file; a line ending taken from the machine would make a notebook
        // saved on one machine differ from the same notebook saved on another.
        var text = Verbs().ReadStep("""{"step": "read.csv", "path": "titanic.csv"}""").AsBlockText();

        Assert.Contains('\n', text);
        Assert.DoesNotContain('\r', text);
    }

    [Fact]
    public void EveryPartCarriesItsOwnIdUnderTheOwnersName_AndVersoTakesEachOne()
    {
        // An id written in two places drifts into two ids, and a mismatch shows up as nothing happening. Each part
        // says its id once, under the name its author owns, and Verso's own validation accepts every part.
        var version = typeof(StepCellType).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!
            .InformationalVersion.Split('+')[0];
        var parts = Parts().Select(type => (IExtension)Activator.CreateInstance(type)!).ToArray();

        Assert.NotEmpty(parts);
        Assert.All(Parts(), type => Assert.True(type.IsPublic && type.IsSealed, type.Name));
        Assert.All(parts, part => Assert.StartsWith("io.github.xkqg.deepsharp.notebooks.", part.ExtensionId, StringComparison.Ordinal));
        Assert.Equal(parts.Length, parts.Select(part => part.ExtensionId).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.All(parts, part => Assert.Empty(new ExtensionHost().ValidateExtension(part)));
        Assert.All(parts, part => Assert.Equal(version, part.Version));
        Assert.All(parts, part => Assert.Equal("H.P. Gansevoort", part.Author));
        Assert.All(parts, part => Assert.False(string.IsNullOrWhiteSpace(part.Name) || string.IsNullOrWhiteSpace(part.Description), part.ExtensionId));
        Assert.All(parts, part => Assert.Equal(part.ExtensionId, part.GetType().GetField("Id")!.GetRawConstantValue()));
    }

    [Fact]
    public async Task AnInstalledPackageIsFoundByVersosOwnDiscovery()
    {
        await using var notebook = await Notebook.OpenAsync();

        Assert.Contains(notebook.Host.GetCellTypes(), type => type.CellTypeId == StepCellType.StepType);
        Assert.Contains(notebook.Host.GetKernels(), kernel => kernel.LanguageId == StepKernel.Language);
        Assert.Contains(notebook.Host.GetRenderers(), renderer => renderer.ExtensionId == StepRenderer.Id);
        Assert.Equal(StepKernel.Language, notebook.AddBlock(new StepCellType().GetDefaultContent()).Language);
    }

    [Fact]
    public async Task RunningABlock_SaysWhichStageItIsAndWhatItDoes_WhileItRuns()
    {
        await using var notebook = await Notebook.OpenAsync();
        var cell = notebook.AddBlock("""{"step": "read.csv", "path": "titanic.csv"}""");

        var card = Assert.Single(await notebook.RunAsync(cell));

        Assert.Equal("text/html", card.MimeType);
        Assert.False(card.IsError);
        Assert.Contains("read.csv", card.Content, StringComparison.Ordinal);
        Assert.Contains(">read<", card.Content, StringComparison.Ordinal);
        Assert.Contains(Verbs().Describe("read.csv").Purpose, card.Content, StringComparison.Ordinal);

        // Written while it ran, so the front end is told at once rather than when the run ends.
        Assert.Contains(cell.Id, notebook.Pushed);
    }

    [Theory]
    [InlineData("""{"step": "read.csv"}""", "(1,1)", "path")]
    [InlineData("{\"step\": \"read.csv\", \"path\": \"x.csv\"}\n{\"step\": \"read.csv\", \"path\": \"y.csv\"}", "(2,1)", "JSON")]
    [InlineData("""[{"step": "read.csv", "path": "x.csv"}]""", "(1,1)", "object")]
    [InlineData("""{"step": "read.cvs", "path": "x.csv"}""", "(1,1)", "read.cvs")]
    [InlineData("", "(1,1)", "JSON")]
    [InlineData("""{"step": "normalise", "column": "a", "scale": "sideways", "outOfRange": "pass"}""", "(1,1)", "sideways")]
    public async Task RunningABlockThatIsNotOneStep_SaysWhatIsWrongAndWhere(string source, string place, string word)
    {
        await using var notebook = await Notebook.OpenAsync();
        var cell = notebook.AddBlock(source);

        var card = Assert.Single(await notebook.RunAsync(cell));

        Assert.True(card.IsError);
        Assert.Contains(place, card.Content, StringComparison.Ordinal);
        Assert.Contains(word, card.Content, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("read.csv", "read")]
    [InlineData("declare", "columns")]
    [InlineData("order.by", "order")]
    [InlineData("feature.add", "features")]
    [InlineData("feature.indicator", "features")]
    [InlineData("drop.gaps", "clean")]
    [InlineData("drop.columns", "clean")]
    [InlineData("split.stratified", "split")]
    [InlineData("normalise", "learned from the training rows")]
    [InlineData("evidence.profile", "evidence")]
    [InlineData("target", "target")]
    public async Task TheStageABlockBelongsTo_FollowsWhatItsStepDoes(string verb, string stage)
    {
        // Worked out from the capability the step acts through, never stored beside it, so it cannot drift from
        // what the step does — a verb another package brings included.
        await using var notebook = await Notebook.OpenAsync();
        var cell = notebook.AddBlock(Verbs().Describe(verb).Template);

        var card = Assert.Single(await notebook.RunAsync(cell));

        Assert.Contains($">{stage}<", card.Content, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("read.csv", typeof(IOpensRows))]
    [InlineData("declare", typeof(IBindsColumns))]
    [InlineData("order.by", typeof(IOrdersRows))]
    [InlineData("feature.add", typeof(IAddsColumns))]
    [InlineData("feature.indicator", typeof(IAddsColumns))]
    [InlineData("drop.gaps", typeof(IDropsRows))]
    [InlineData("drop.columns", typeof(IDropsColumns))]
    [InlineData("split.stratified", typeof(ISplitStep))]
    [InlineData("normalise", typeof(IFittedStep))]
    [InlineData("evidence.profile", typeof(IProducesEvidence))]
    [InlineData("target", typeof(TargetStep))]
    public void EveryStep_ActsThroughOneCapability_ThatItsStageIsNamedAfter(string verb, Type capability)
    {
        // The capability decides what a step can be swapped for; the stage is only its name, and two capabilities
        // may share one — leaving out rows and leaving out a column are both cleaning.
        var step = Verbs().ReadStep(Verbs().Describe(verb).Template);

        Assert.Equal(capability, step.ActingCapability());
        Assert.False(string.IsNullOrWhiteSpace(step.Stage()));
    }

    [Fact]
    public void EveryPublicConstantOfAPart_IsAnIdOrAKeyOfItsContract()
    {
        // What a part publishes is what another program relies on: its id, the language and block type Verso
        // stores, and the keys a C# cell reads. A word a button or a form uses is the package's own business.
        string[] contract = ["Id", "Language", "StepType", "HandOver", "Folder"];
        var constants = Parts()
            .SelectMany(type => type.GetFields(BindingFlags.Public | BindingFlags.Static).Where(field => field.IsLiteral))
            .Select(field => $"{field.DeclaringType!.Name}.{field.Name}");

        Assert.All(constants, constant => Assert.Contains(constant.Split('.')[1], contract));
    }

    [Fact]
    public async Task WhatVersoNeverHandsAKernel_IsRefused_NotReadAsAnEmptyBlock()
    {
        var kernel = new StepKernel();

        await Assert.ThrowsAsync<ArgumentNullException>(() => kernel.ExecuteAsync(null!, null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => kernel.GetCompletionsAsync(null!, 0));
        await Assert.ThrowsAsync<ArgumentNullException>(() => kernel.GetDiagnosticsAsync(null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => kernel.GetHoverInfoAsync(null!, 0));
        await Assert.ThrowsAsync<ArgumentNullException>(() => new StepRenderer().RenderInputAsync(null!, null!));
    }

    [Fact]
    public async Task EveryKernel_ReachesTheOneSessionOfItsNotebook_WhateverIsSwitchedOff_AndOneVersoNeverLoadedRefuses()
    {
        await using var notebook = await Notebook.OpenAsync();
        var blocks = notebook.Host.GetCellTypes().OfType<StepCellType>().Single();
        var loaded = notebook.Host.GetKernels().OfType<StepKernel>().Single();

        Assert.Same(blocks.Session, ((StepKernel)blocks.Kernel).Session);
        Assert.Same(blocks.Session, loaded.Session);
        Assert.False(string.IsNullOrWhiteSpace(loaded.DisplayName));

        // Switched off, the block type is left out of what Verso lists as enabled; the notebook's session is not.
        await notebook.Host.DisableExtensionAsync(StepCellType.Id);

        Assert.Same(blocks.Session, loaded.Session);

        var alone = Assert.Throws<InvalidOperationException>(() => new StepKernel().Session);

        Assert.Contains("not loaded by Verso", alone.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AtTheStepKey_TheEditorIsOfferedEveryVerb()
    {
        const string code = """{"step": "no""";
        var offered = await new StepKernel().GetCompletionsAsync(code, code.Length);

        Assert.Equal(["normalise", "normalise.row"], offered.Select(completion => completion.InsertText));
        Assert.All(offered, completion => Assert.Equal(Verbs().Describe(completion.InsertText).Purpose, completion.Description));

        const string empty = """{"step": " """;
        Assert.Equal(
            Verbs().Descriptions.Select(description => description.Verb),
            (await new StepKernel().GetCompletionsAsync(empty, empty.Length - 1)).Select(completion => completion.InsertText));
    }

    [Fact]
    public async Task InsideAStep_TheEditorIsOfferedTheKeysItHasNotWritten()
    {
        const string code = """{"step": "normalise", "column": "age", """;

        var offered = await new StepKernel().GetCompletionsAsync(code, code.Length);

        Assert.Equal(["\"scale\": ", "\"outOfRange\": "], offered.Select(completion => completion.InsertText));
        Assert.Equal(["scale", "outOfRange"], offered.Select(completion => completion.DisplayText));

        const string typing = """{"step": "normalise", "sc""";

        Assert.Equal(["scale"], (await new StepKernel().GetCompletionsAsync(typing, typing.Length)).Select(completion => completion.InsertText));
    }

    [Fact]
    public async Task ForAKeyThatTakesAWord_TheEditorIsOfferedItsWords()
    {
        var scale = (OneOfParameter<Scale>)Verbs().Describe("normalise").Parameters.Single(parameter => parameter.Key == "scale");

        const string inside = """{"step": "normalise", "scale": " """;
        const string before = """{"step": "normalise", "scale": """;

        Assert.Equal(scale.Choices, (await new StepKernel().GetCompletionsAsync(inside, inside.Length - 1)).Select(completion => completion.InsertText));
        Assert.Equal(scale.Choices.Select(word => $"\"{word}\""), (await new StepKernel().GetCompletionsAsync(before, before.Length)).Select(completion => completion.InsertText));

        const string parts = """{"step": "feature.timeParts", "parts": [" """;
        var timeParts = (SeveralOfParameter<TimePart>)Verbs().Describe("feature.timeParts").Parameters.Single(parameter => parameter.Key == "parts");

        Assert.Equal(timeParts.Choices, (await new StepKernel().GetCompletionsAsync(parts, parts.Length - 1)).Select(completion => completion.InsertText));

        const string with = """{"step": "fill.missing", "with": " """;
        var strategy = (FillStrategyParameter)Verbs().Describe("fill.missing").Parameters.Single(parameter => parameter.Key == "with");

        // A strategy carrying a number is an object, not a word, so inside the quotes only the words are offered;
        // before them, the object is offered whole.
        Assert.Equal(strategy.Allowed.Where(name => !With.TakesAValue(name)), (await new StepKernel().GetCompletionsAsync(with, with.Length - 1)).Select(completion => completion.InsertText));

        const string strategyValue = """{"step": "fill.missing", "with": """;

        Assert.Contains("{\"kind\": \"constant\", \"value\": 0}", (await new StepKernel().GetCompletionsAsync(strategyValue, strategyValue.Length)).Select(completion => completion.InsertText));

        const string flag = """{"step": "feature.timeParts", "asCategories": """;

        Assert.Equal(["true", "false"], (await new StepKernel().GetCompletionsAsync(flag, flag.Length)).Select(completion => completion.InsertText));
    }

    [Theory]
    [InlineData("""{"step": "normalise", "column": 1""", 33)]
    [InlineData("""{"step": "nowhere", """, 20)]
    [InlineData("""{"step": "normalise", "column": " """, 33)]
    [InlineData("""{"step": "normalise", "colour": " """, 33)]
    [InlineData("""{"step": "normalise"}""", 21)]
    [InlineData("""{"step": "normalise"}""", 99)]
    [InlineData("""{"step": "normalise"}""", -1)]
    [InlineData("""{"step": "declare", "columns": [{"name": " """, 43)]
    public async Task WhereNothingFits_NothingIsOffered(string code, int cursor)
    {
        Assert.Empty(await new StepKernel().GetCompletionsAsync(code, cursor));
    }

    [Fact]
    public async Task EveryKeyOfEveryVerb_IsOfferedWhatItsKindTakes_AndNothingWhereItTakesAnythingAtAll()
    {
        // Words where the kind has words — one of a set, true or false, a way to fill — and nothing where a value is
        // a person's own: a name, a number, a path. A column is known only once a gesture has shown the notebook's
        // pipeline, which a kernel on its own has never seen.
        var kernel = new StepKernel();

        foreach (var description in Verbs().Descriptions)
        {
            foreach (var parameter in description.Parameters)
            {
                foreach (var key in parameter.Keys)
                {
                    var code = $$"""{"step": "{{description.Verb}}", "{{key}}": """;
                    var offered = await kernel.GetCompletionsAsync(code, code.Length);
                    var hasWords = parameter.GetType().Name is "OneOfParameter`1" or "TrueOrFalseParameter" or "FillStrategyParameter";

                    Assert.True(hasWords == offered.Count > 0, $"{description.Verb}.{key}: {offered.Count}");
                }
            }
        }
    }

    [Fact]
    public async Task ABlockIsDrawnAsWhatItHolds_AndItsRunAsWhatTheRunWrote()
    {
        // The step stays in view after a run, because it is what a person edits; what the run wrote is shown as
        // it was written.
        var renderer = new StepRenderer();
        var card = CellOutput.Html("<p>the card</p>");

        Assert.Equal(StepCellType.StepType, renderer.CellTypeId);
        Assert.False(string.IsNullOrWhiteSpace(renderer.DisplayName));
        Assert.False(renderer.CollapsesInputOnExecute);
        Assert.Equal(CellVisibilityHint.Content, renderer.DefaultVisibility);
        Assert.Equal("json", renderer.GetEditorLanguage());
        Assert.Equal(new RenderResult("text/plain", "{}"), await renderer.RenderInputAsync("{}", null!));
        Assert.Equal(new RenderResult("text/html", "<p>the card</p>"), await renderer.RenderOutputAsync(card, null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => renderer.RenderOutputAsync(null!, null!));
    }

    [Fact]
    public async Task HoveringOverAKeyOrAVerb_SaysWhatItMeans()
    {
        const string code = """{"step": "normalise", "scale": "minMax"}""";
        var kernel = new StepKernel();
        var scale = Verbs().Describe("normalise").Parameters.Single(parameter => parameter.Key == "scale");

        Assert.Equal(scale.Description, (await kernel.GetHoverInfoAsync(code, code.IndexOf("scale", StringComparison.Ordinal) + 2))!.Content);
        Assert.Equal(Verbs().Describe("normalise").Purpose, (await kernel.GetHoverInfoAsync(code, code.IndexOf("normalise", StringComparison.Ordinal) + 1))!.Content);
        Assert.Null(await kernel.GetHoverInfoAsync(code, code.IndexOf("minMax", StringComparison.Ordinal) + 1));
        Assert.Null(await kernel.GetHoverInfoAsync(code, 0));
        Assert.Null(await kernel.GetHoverInfoAsync("""{"step": "nowhere", "a": 1}""", 3));
        Assert.Null(await kernel.GetHoverInfoAsync("""{"step": "normalise", "colour": 1}""", 25));
    }

    [Fact]
    public async Task TheFaultsOfABlock_AreItsDiagnosticsToo_CountedFromOne()
    {
        var kernel = new StepKernel();

        var fault = Assert.Single(await kernel.GetDiagnosticsAsync("""{"step": "read.csv"}"""));

        Assert.Equal(DiagnosticSeverity.Error, fault.Severity);
        Assert.Equal(1, fault.StartLine);
        Assert.Equal(1, fault.StartColumn);
        Assert.Contains("path", fault.Message, StringComparison.Ordinal);
        Assert.Empty(await kernel.GetDiagnosticsAsync("""{"step": "read.csv", "path": "x.csv"}"""));
    }
}
